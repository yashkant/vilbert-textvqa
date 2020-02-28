# Copyright (c) Facebook, Inc. and its affiliates.
import functools
import math
import torch
import logging
from torch import nn
import torch.nn.functional as F
from pytorch_transformers.modeling_bert import (
    BertLayerNorm, BertEmbeddings, BertEncoder, BertConfig,
    BertPreTrainedModel
)
from tools.registry import registry
from vilbert.textvqa_encoders import ImageEncoder
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



class MMT_VQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.image_embeddings = BertImageEmbeddings(config)
        self.encoder = BertEncoder(config)
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


        # build embeddings for predictions in previous decoding steps
        # fixed_ans_emb is an embedding lookup table for each fixed vocabulary
        # dec_emb = self.prev_pred_embeddings(fixed_ans_emb, batch_dict["ocr_mmt_in"], batch_dict["train_prev_inds"])

        # a zero mask for decoding steps, so the encoding steps elements can't
        # attend to decoding steps.
        # A triangular causal mask will be filled for the decoding steps
        # later in extended_attention_mask
        # dec_mask = torch.zeros(
        #     dec_emb.size(0),
        #     dec_emb.size(1),
        #     dtype=torch.long,
        #     device=dec_emb.device
        # )
        # encoder_inputs = torch.cat(
        #     [batch_dict["text_bert_emb"], batch_dict["obj_mmt_in"], batch_dict["ocr_mmt_in"], dec_emb],
        #     dim=1
        # )
        # attention_mask = torch.cat(
        #     [batch_dict["question_mask"], batch_dict["pad_obj_mask"], batch_dict["pad_ocr_mask"], dec_mask],
        #     dim=1
        # )
        #
        # # offsets of each modality in the joint embedding space
        # txt_max_num = batch_dict["question_mask"].size(-1)
        # obj_max_num = batch_dict["pad_obj_mask"].size(-1)
        # ocr_max_num = batch_dict["pad_ocr_mask"].size(-1)
        # dec_max_num = dec_mask.size(-1)
        # txt_begin = 0
        # txt_end = txt_begin + txt_max_num
        # ocr_begin = txt_max_num + obj_max_num
        # ocr_end = ocr_begin + ocr_max_num

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to
        # [batch_size, num_heads, from_seq_length, to_seq_length]
        # to_seq_length = attention_mask.size(1)
        # from_seq_length = to_seq_length

        # generate the attention mask similar to prefix LM
        # all elements can attend to the elements in encoding steps
        extended_attention_mask = joint_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility

        # # decoding step elements can attend to themselves in a causal manner
        # extended_attention_mask[:, :, -dec_max_num:, -dec_max_num:] = \
        #     _get_causal_mask(dec_max_num, encoder_inputs.device)
        #
        # flip the mask, so that invalid attention pairs have -10000.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        assert not extended_attention_mask.requires_grad
        head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(
            joint_embeddings,
            extended_attention_mask,
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
