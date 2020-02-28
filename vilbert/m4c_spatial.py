# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
import logging
from torch import nn
import torch.nn.functional as F
from collections import Counter
from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)

from vilbert.vilbert import GeLU

logger = logging.getLogger(__name__)


class M4C(nn.Module):
    """
    M4C has two transfomers MMT and TextBert.
    """

    def __init__(self, mmt_config, text_bert_config):
        super().__init__()
        # self.mmt_config = BertConfig(**self.config.mmt)
        self.mmt_config = mmt_config
        self.text_bert_config = text_bert_config

        if self.mmt_config.finetune_ocr_obj:
            logger.info("Finetuning object and ocr FRCN layer")
            self.frcn_encoder_type = "finetune_faster_rcnn_fpn_fc7"
        else:
            logger.info("Not finetuning object and ocr FRCN layer")
            self.frcn_encoder_type = "default"

        self.normalize = self.mmt_config.normalize
        if not self.mmt_config.normalize:
            logger.info("Not normalizing OCR and Object features")

        self.fusion_method = self.mmt_config.fusion_method
        logger.info(f"Fusion Method is : {self.fusion_method}")

        # build the models
        self.build()

    def build(self):
        # modules requiring custom learning rates (usually for finetuning)
        self.finetune_modules = []

        # split model building into several components
        self._build_txt_encoding()
        self._build_mmt()
        self._build_output()


    def _build_txt_encoding(self):
        TEXT_BERT_HIDDEN_SIZE = 768
        # self.text_bert_config = BertConfig(**self.config.text_bert)
        if self.text_bert_config.text_bert_init_from_bert_base:
            self.text_bert = TextBert.from_pretrained(
                'bert-base-uncased', config=self.text_bert_config
            )
            # Use a smaller learning rate on text bert when initializing
            # from BERT_BASE
            self.finetune_modules.append({
                'module': self.text_bert,
                'lr_scale': self.text_bert_config.lr_scale_text_bert,
            })
        else:
            logger.info('NOT initializing text_bert from BERT_BASE')
            self.text_bert = TextBert(self.text_bert_config)

        # if the text bert output dimension doesn't match the
        # multimodal transformer (mmt) hidden dimension,
        # add a linear projection layer between the two
        if self.mmt_config.hidden_size != TEXT_BERT_HIDDEN_SIZE:
            logger.info(
                'Projecting text_bert output to {} dim'.format(
                    self.mmt_config.hidden_size
                )
            )
            self.text_bert_out_linear = nn.Linear(
                TEXT_BERT_HIDDEN_SIZE, self.mmt_config.hidden_size
            )
        else:
            self.text_bert_out_linear = nn.Identity()


    def _build_mmt(self):
        self.mmt = MMT_VQA(self.mmt_config)

        # allow specifying a different/scaled lr for multimodal transformer
        self.finetune_modules.append({
            'module': self.mmt,
            'lr_scale': self.mmt_config.lr_scale_mmt,
        })

    def _build_output(self):
        # dynamic OCR-copying scores with pointer network
        self.vil_prediction = SimpleClassifier(
            self.mmt_config.hidden_size, self.mmt_config.hidden_size * 2, 3129, 0.5
        )
        self.vil_prediction_gqa = SimpleClassifier(
            self.mmt_config.hidden_size, self.mmt_config.hidden_size * 2, 1533, 0.5
        )
        self.dropout = nn.Dropout(self.mmt_config.hidden_dropout_prob)


    def _build_aux_heads(self):
        from vilbert.vilbert import SimpleClassifier
        # spatial-category classification head
        self.origin_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32)
        self.dest_transform = SimpleClassifier(self.mmt_config.hidden_size, 128, 32)
        self.spatial_classifier = nn.Linear(32, 12)

    def forward(self, batch_dict):
        self._forward_mmt_and_text(batch_dict)
        self._forward_output(batch_dict)
        results_dict = {
            "vil_prediction": batch_dict["vil_prediction"],
        }
        return results_dict

    def _forward_mmt_and_text(self, batch_dict):
        # first forward the text BERT layers
        text_bert_out = self.text_bert(batch_dict)
        batch_dict['text_bert_emb'] = self.text_bert_out_linear(text_bert_out)

        mmt_results = self.mmt(
            batch_dict,
        )
        batch_dict.update(mmt_results)

    def _forward_output(self, batch_dict):
        if self.fusion_method == "sum":
            batch_dict["pooled_output"] = self.dropout(batch_dict["pooled_text_output"] + batch_dict["pooled_image_output"])
        elif self.fusion_method == "mul":
            batch_dict["pooled_output"] = self.dropout(batch_dict["pooled_text_output"] * batch_dict["pooled_image_output"])
        else:
            assert False
        batch_dict["vil_prediction"] = self.vil_prediction(batch_dict["pooled_output"])
        # batch_dict["vil_prediction_gqa"] = self.vil_prediction_gqa(batch_dict["pooled_output"])

    def get_optimizer_parameters(self, base_lr):
        optimizer_param_groups = []

        # base_lr = config.optimizer_attributes.params.lr
        # collect all the parameters that need different/scaled lr
        finetune_params_set = set()
        for m in self.finetune_modules:
            optimizer_param_groups.append({
                "params": list(m['module'].parameters()),
                "lr": base_lr * m['lr_scale']
            })
            finetune_params_set.update(list(m['module'].parameters()))
        # remaining_params are those parameters w/ default lr
        remaining_params = [
             p for p in self.parameters() if p not in finetune_params_set
        ]
        # put the default lr parameters at the beginning
        # so that the printed lr (of group 0) matches the default lr
        optimizer_param_groups.insert(0, {"params": remaining_params})

        return optimizer_param_groups


class TextBert(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, batch_dict):
        encoder_inputs = self.embeddings(batch_dict["question_indices"])
        attention_mask = batch_dict["question_mask"]

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            encoder_inputs,
            extended_attention_mask,
            head_mask=head_mask
        )
        seq_output = encoder_outputs[0]

        return seq_output


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """
    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()
        self.image_embeddings = nn.Linear(2048, config.hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.hidden_size)
        self.image_type_embeddings = nn.Embedding(1, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)
        type_ids = input_ids.new_zeros(img_embeddings.shape[:-1], dtype=torch.long)
        img_type_embeddings = self.image_type_embeddings(type_ids)
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings + img_type_embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SpatialBertSelfAttention(nn.Module):
    """
    Todo: Keep 768 and build zero-mask for 12th head (not needed with identity relation)
    """

    def __init__(self, config, use_implicit=False):
        super(SpatialBertSelfAttention, self).__init__()
        assert hasattr(config, "num_spatial_relations")

        self.num_attention_heads = config.num_spatial_relations
        self.num_spatial_relations = config.num_spatial_relations

        if hasattr(config, "num_implicit_relations") and use_implicit:
            self.num_attention_heads += config.num_implicit_relations
            self.num_implicit_relations = config.num_implicit_relations

        if config.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, self.num_attention_heads))

        self.output_attentions = config.output_attentions
        self.max_seq_len = config.max_seq_length
        self.mask_quadrants = config.attention_mask_quadrants

        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        if hasattr(config, "no_drop") and config.no_drop:
            logger.info("not using dropout")
            self.dropout = nn.Dropout(0.0)

        self.use_bias = False
        if hasattr(config, "use_bias") and config.use_bias:
            self.use_bias = True
            logger.info("using head biases")
            self.biases = torch.nn.Embedding(1, config.hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, spatial_adj_matrix, head_mask=None):
        """
        spatial_adj_matrix: (bs, num_ocr + num_obj, num_ocr + num_obj, type_relations == num_heads)
        TODO: Is there way you can draw some connections across these heads? Like one-head using some
            other heads K/Q/V, can this modelled as any cross-relationship, think think.

        One Problem: We concatenate the outputs from all the 12 heads, now:
            - Leave others as zeros? (this will happen if there's no rel_i th of this embedding with others) (as it is) [rel_enc1, rel_enc2, ... rel_enc11]
            - Balancing these features might be an issue if I have these zeros popping in at places.
            - Also balancing them even when they aren't zero? Like you had 12 heads contributing before,
                but now the heads might be anywhere from 1-12.

        spatial_adj_matrix has 0s across the diagonal terms, attaching an identity matrix for this

        """
        # build attention-mask from spatial_adj_matrix
        batch_size, ocr_obj_num, _, num_spatial_heads = spatial_adj_matrix.shape
        num_features = hidden_states.size(1)

        # Removing masking all quadrants
        spatial_attention_mask = attention_mask.new_ones((batch_size, num_features, num_features, num_spatial_heads))

        # Add explicit mask
        spatial_attention_mask[:, self.max_seq_len:self.max_seq_len + ocr_obj_num,
        self.max_seq_len:self.max_seq_len + ocr_obj_num, :] = spatial_adj_matrix

        # Add implicit mask
        if self.num_attention_heads != self.num_spatial_relations:
            assert hasattr(self, "num_implicit_relations")
            implicit_attention_mask = attention_mask.new_ones(
                (batch_size, num_features, num_features, self.num_implicit_relations))
            spatial_attention_mask = torch.cat([spatial_attention_mask, implicit_attention_mask], dim=-1)

        assert spatial_attention_mask.shape == (batch_size, num_features, num_features, self.num_attention_heads)

        # Mask attention-quadrants (spatial relations only)
        for quadrant in self.mask_quadrants:
            if quadrant == 1:
                spatial_attention_mask[:, :self.max_seq_len, :self.max_seq_len, :self.num_spatial_relations] = 0
            elif quadrant == 2:
                spatial_attention_mask[:, :self.max_seq_len, self.max_seq_len:self.max_seq_len + ocr_obj_num,
                :self.num_spatial_relations] = 0
            elif quadrant == 4:
                spatial_attention_mask[:, self.max_seq_len:self.max_seq_len + ocr_obj_num, :self.max_seq_len,
                :self.num_spatial_relations] = 0
            elif quadrant == 7:
                spatial_attention_mask[:, self.max_seq_len + ocr_obj_num:, :self.max_seq_len,
                :self.num_spatial_relations] = 0
            elif quadrant == 8:
                spatial_attention_mask[:, self.max_seq_len + ocr_obj_num:,
                self.max_seq_len:self.max_seq_len + ocr_obj_num, :self.num_spatial_relations] = 0
            elif quadrant == 9:
                spatial_attention_mask[:, self.max_seq_len + ocr_obj_num:, self.max_seq_len + ocr_obj_num:,
                :self.num_spatial_relations] = 0
            else:
                raise ValueError

        spatial_attention_mask = (1.0 - spatial_attention_mask) * -10000.0
        spatial_attention_mask = spatial_attention_mask.permute((0, 3, 1, 2))

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Prevent imbalanced masking
        combined_mask = torch.min(attention_mask, spatial_attention_mask)
        assert len(torch.unique(combined_mask)) == 2

        # for entities that are totally masked
        entity_probs_mask = (combined_mask.max(dim=-1)[0] + 10000.0) / 10000.0
        entity_probs_mask = entity_probs_mask.unsqueeze(-1)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + combined_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # zero-out completely masked entities
        attention_probs = attention_probs * entity_probs_mask

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.use_bias:
            context_layer = context_layer + self.biases(context_layer.new_zeros(1).long())

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class SpatialBertAttention(nn.Module):
    def __init__(self, config, use_implicit=False):
        super(SpatialBertAttention, self).__init__()
        self.self = SpatialBertSelfAttention(config, use_implicit)
        from pytorch_transformers.modeling_bert import BertSelfOutput
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_tensor, attention_mask, spatial_adj_matrix, head_mask=None):
        self_outputs = self.self(input_tensor, attention_mask, spatial_adj_matrix, head_mask)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SpatialBertLayer(nn.Module):
    def __init__(self, config, use_implicit=False):
        super(SpatialBertLayer, self).__init__()
        from pytorch_transformers.modeling_bert import BertIntermediate, BertOutput
        self.attention = SpatialBertAttention(config, use_implicit)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, spatial_adj_matrix, head_mask=None):
        attention_outputs = self.attention(hidden_states, attention_mask, spatial_adj_matrix, head_mask)
        attention_output = attention_outputs[0]
        # Intermediate is dense + activation
        intermediate_output = self.intermediate(attention_output)
        # Output is dense + dropout + residual + layernorm
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class BertSpatialEncoder(nn.Module):
    def __init__(self, config):
        super(BertSpatialEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        from pytorch_transformers.modeling_bert import BertLayer

        # backward compatibility
        if config.layer_type_list is None:
            logger.info("layer_type_list not passed, generating it!")
            self.layer_type_list = ["n"] * config.num_hidden_layers

            if hasattr(config, "num_implicit_relations") and config.num_implicit_relations != 0:
                spatial_type = ["i"]
            else:
                spatial_type = ["s"]

            if config.spatial_type == "bottom":
                self.layer_type_list = spatial_type * config.num_spatial_layers + self.layer_type_list
            if config.spatial_type == "top":
                self.layer_type_list = self.layer_type_list + spatial_type * config.num_spatial_layers
        else:
            self.layer_type_list = config.layer_type_list

        logger.info(f"Layers type list is: {self.layer_type_list}")
        counter = Counter(self.layer_type_list)

        self.num_spatial_layers = counter["s"]
        self.num_normal_layers = counter["n"]
        self.num_implicit_layers = counter["i"]

        logger.info(f"Num Spatial Layers: {self.num_spatial_layers}")
        logger.info(f"Num Normal Layers: {self.num_normal_layers}")
        logger.info(f"Num Implicit Layers: {self.num_implicit_layers}")

        if config.mix_list is None:
            self.mix_list = ["none"] * len(self.layer_type_list)
        else:
            self.mix_list = config.mix_list
        assert len(self.mix_list) == len(self.layer_type_list)
        logger.info(f"Mix list: {self.mix_list}")

        self.normal_layers = nn.ModuleList([BertLayer(config) for _ in range(self.num_normal_layers)])
        self.spatial_layers = nn.ModuleList([SpatialBertLayer(config) for _ in range(self.num_spatial_layers)])
        self.implicit_layers = nn.ModuleList([SpatialBertLayer(config, True) for _ in range(self.num_implicit_layers)])

    def forward(self, hidden_states, attention_mask, batch_dict, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        normal_iter = iter(self.normal_layers)
        spatial_iter = iter(self.spatial_layers)
        implicit_iter = iter(self.implicit_layers)

        for layer_type, mix_type in zip(self.layer_type_list, self.mix_list):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if layer_type == "n":
                layer_module = next(normal_iter)
                layer_outputs = layer_module(hidden_states, attention_mask)
            elif layer_type == "s":
                layer_module = next(spatial_iter)
                if mix_type == "share3":
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix_share3"]
                elif mix_type == "quad4":
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix_quad4"]
                else:
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix"]
                layer_outputs = layer_module(hidden_states, attention_mask, spatial_adj_matrix)
            elif layer_type == "i":
                layer_module = next(implicit_iter)
                if mix_type == "share3":
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix_share3"]
                elif mix_type == "quad4":
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix_quad4"]
                else:
                    spatial_adj_matrix = batch_dict["spatial_adj_matrix"]
                layer_outputs = layer_module(hidden_states, attention_mask, spatial_adj_matrix)
            else:
                raise ValueError
            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        assert next(normal_iter, None) is None
        assert next(spatial_iter, None) is None
        assert next(implicit_iter, None) is None

        # # re-arrange spatial-layers if needed
        # layers = [("normal", self.layer), ("spatial", self.spatial_layer)]
        # if self.spatial_type == "bottom":
        #     layers = [layers[1], layers[0]]
        # else:
        #     assert self.spatial_type == "top"
        #
        # # Run through layers
        # for layer_type, layers_set in layers:
        #     for i, layer_module in enumerate(layers_set):
        #         if self.output_hidden_states:
        #             all_hidden_states = all_hidden_states + (hidden_states,)
        #
        #         if layer_type == "normal":
        #             layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i])
        #         elif layer_type == "spatial":
        #             layer_outputs = layer_module(hidden_states, attention_mask, spatial_adj_matrix)
        #         else:
        #             raise ValueError
        #
        #         hidden_states = layer_outputs[0]
        #
        #         if self.output_attentions:
        #             all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class MMT_VQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.image_embeddings = BertImageEmbeddings(config)
        self.encoder = BertSpatialEncoder(config)
        self.text_pooler = BertTextPooler(config)
        self.image_pooler = BertImagePooler(config)

        # self.apply(self.init_weights)  # old versions of pytorch_transformers
        self.init_weights()

    def forward(self, batch_dict):

        text_embeddings = batch_dict["text_bert_emb"]
        image_embeddings = self.image_embeddings(
            batch_dict["input_imgs"],
            batch_dict["image_loc"]
        )

        joint_embeddings = torch.cat([text_embeddings, image_embeddings], dim=1)
        joint_mask = torch.cat([batch_dict["question_mask"], batch_dict["image_mask"]], dim=-1)
        extended_attention_mask = joint_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            joint_embeddings,
            extended_attention_mask,
            batch_dict,
            head_mask=head_mask
        )

        text_len = text_embeddings.shape[1]
        pooled_text_output = self.text_pooler(encoder_outputs[0][:, :text_len])
        pooled_image_output = self.image_pooler(encoder_outputs[0][:, text_len:])

        results = {
            'pooled_text_output': pooled_text_output,
            'pooled_image_output': pooled_image_output,
        }
        return results


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def _get_mask(nums, max_num):
    # non_pad_mask: b x lq, torch.float32, 0. on PAD
    batch_size = nums.size(0)
    arange = torch.arange(0, max_num).unsqueeze(0).expand(batch_size, -1)
    non_pad_mask = arange.to(nums.device).lt(nums.unsqueeze(-1))
    non_pad_mask = non_pad_mask.type(torch.float32)
    return non_pad_mask


@functools.lru_cache(maxsize=32)
def _get_causal_mask(seq_length, device):
    # generate a lower triangular mask
    mask = torch.zeros(seq_length, seq_length, device=device)
    for i in range(seq_length):
        for j in range(i + 1):
            mask[i, j] = 1.
    return mask


def _batch_gather(x, inds):
    assert x.dim() == 3
    batch_size = x.size(0)
    length = x.size(1)
    dim = x.size(2)
    x_flat = x.view(batch_size * length, dim)

    batch_offsets = torch.arange(batch_size, device=inds.device) * length
    batch_offsets = batch_offsets.unsqueeze(-1)
    assert batch_offsets.dim() == inds.dim()
    inds_flat = batch_offsets + inds
    results = F.embedding(inds_flat, x_flat)
    return results


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)
