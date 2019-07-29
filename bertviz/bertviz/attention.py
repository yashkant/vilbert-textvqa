import torch
from collections import defaultdict
import pdb

def get_attention_xlnet(model, tokenizer, text):

    """Compute representation of the attention from xlnet to pass to the d3 visualization

    Args:
      model: xlnet model
      tokenizer: xlnet tokenizer
      text: Input text

    Returns:
      Dictionary of attn representations with the structure:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape (num_heads, source_seq_len, target_seq_len)
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """

    # Prepare inputs to model

    # From https://github.com/zihangdai/xlnet/blob/master/classifier_utils.py
    seg_id_a = 0
    seg_id_cls = 2
    seg_id_sep = 3

    # Based on https://github.com/zihangdai/xlnet/blob/master/data_utils.py
    cls_token = '<cls>'
    sep_token = '<sep>'
    tokens = []
    segment_ids = []
    for token in tokenizer.tokenize(text):
        tokens.append(token)
        segment_ids.append(seg_id_a)
    tokens.append(sep_token)
    segment_ids.append(seg_id_sep)
    tokens.append(cls_token)
    segment_ids.append(seg_id_cls)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor([segment_ids])

    # Call model to get attention data
    model.eval()
    _, _, attn_data_list = model(tokens_tensor, token_type_tensor)

    # Format attention data for visualization
    all_attns = []
    for layer, attn_data in enumerate(attn_data_list):
        attn = attn_data['attn'][0]  # assume batch_size=1; output shape = (num_heads, seq_len, seq_len)
        all_attns.append(attn.tolist())
    results = {
        'attn': all_attns,
        'left_text': tokens,
        'right_text': tokens
    }
    return {'all': results}

def get_attention_bert(model, tokenizer, sentence_a, sentence_b, include_queries_and_keys=False):

    """Compute representation of the attention for BERT to pass to the d3 visualization

    Args:
      model: BERT model
      tokenizer: BERT tokenizer
      sentence_a: Sentence A string
      sentence_b: Sentence B string
      include_queries_and_keys: Indicates whether to include queries/keys in results

    Returns:
      Dictionary of attn representations with the structure:
      {
        'all': All attention (source = AB, target = AB)
        'aa': Sentence A self-attention (source = A, target = A)
        'bb': Sentence B self-attention (source = B, target = B)
        'ab': Sentence A -> Sentence B attention (source = A, target = B)
        'ba': Sentence B -> Sentence A attention (source = B, target = A)
      }
      where each value is a dictionary:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """

    # Prepare inputs to model
    tokens_a = ['[CLS]'] + tokenizer.tokenize(sentence_a)  + ['[SEP]']
    tokens_b = tokenizer.tokenize(sentence_b) + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(tokens_a + tokens_b)
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor([[0] * len(tokens_a) + [1] * len(tokens_b)])

    # Call model to get attention data
    model.eval()
    _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)

    # Populate map with attn data and, optionally, query, key data
    keys_dict = defaultdict(list)
    queries_dict = defaultdict(list)
    attn_dict = defaultdict(list)
    slice_a = slice(0, len(tokens_a))  # Positions corresponding to sentence A in input
    slice_b = slice(len(tokens_a), len(tokens_a) + len(tokens_b))  # Position corresponding to sentence B in input
    for layer, attn_data in enumerate(attn_data_list):
        # Process attention
        attn = attn_data['attn'][0]  # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        attn_dict['all'].append(attn.tolist())
        attn_dict['aa'].append(attn[:, slice_a, slice_a].tolist())  # Append A->A attention for layer, across all heads
        attn_dict['bb'].append(attn[:, slice_b, slice_b].tolist())  # Append B->B attention for layer, across all heads
        attn_dict['ab'].append(attn[:, slice_a, slice_b].tolist())  # Append A->B attention for layer, across all heads
        attn_dict['ba'].append(attn[:, slice_b, slice_a].tolist())  # Append B->A attention for layer, across all heads
        
        # Process queries and keys
        if include_queries_and_keys:
            queries = attn_data['queries'][0]  # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            keys = attn_data['keys'][0]  # assume batch_size=1; shape = [num_heads, seq_len, vector_size]
            queries_dict['all'].append(queries.tolist())
            keys_dict['all'].append(keys.tolist())
            queries_dict['a'].append(queries[:, slice_a, :].tolist())
            keys_dict['a'].append(keys[:, slice_a, :].tolist())
            queries_dict['b'].append(queries[:, slice_b, :].tolist())
            keys_dict['b'].append(keys[:, slice_b, :].tolist())

    results = {
        'all': {
            'attn': attn_dict['all'],
            'left_text': tokens_a + tokens_b,
            'right_text': tokens_a + tokens_b
        },
        'aa': {
            'attn': attn_dict['aa'],
            'left_text': tokens_a,
            'right_text': tokens_a
        },
        'bb': {
            'attn': attn_dict['bb'],
            'left_text': tokens_b,
            'right_text': tokens_b
        },
        'ab': {
            'attn': attn_dict['ab'],
            'left_text': tokens_a,
            'right_text': tokens_b
        },
        'ba': {
            'attn': attn_dict['ba'],
            'left_text': tokens_b,
            'right_text': tokens_a
        }
    }
    if include_queries_and_keys:
        results['all'].update({
            'queries': queries_dict['all'],
            'keys': keys_dict['all'],
        })
        results['aa'].update({
            'queries': queries_dict['a'],
            'keys': keys_dict['a'],
        })
        results['bb'].update({
            'queries': queries_dict['b'],
            'keys': keys_dict['b'],
        })
        results['ab'].update({
            'queries': queries_dict['a'],
            'keys': keys_dict['b'],
        })
        results['ba'].update({
            'queries': queries_dict['b'],
            'keys': keys_dict['a'],
        })
    return results

def get_attention_gpt2(model, tokenizer, text, include_queries_and_keys=False):

    """Compute representation of the attention from GPT-2 to pass to the d3 visualization

    Args:
      model: GPT-2 model
      tokenizer: GPT-2 tokenizer
      text: Input text
      include_queries_and_keys: Indicates whether to include queries/keys in results

    Returns:
      Dictionary of attn representations with the structure:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape (num_heads, source_seq_len, target_seq_len)
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """

    # Prepare inputs to model
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.decode([t]).strip() for t in token_ids]
    tokens_tensor = torch.tensor([token_ids])

    # Call model to get attention data
    model.eval()
    prediction_scores_t, prediction_scores_v, seq_relationship_score, attn_data_list = model(
        input_ids,
        image_feat,
        image_loc,
        segment_ids,
        input_mask,
        image_mask,
        output_all_attention_masks=True
        )
    
    # _, _, attn_data_list = model(tokens_tensor)

    # Format attention data for visualization
    all_attns = []
    all_queries = []
    all_keys = []
    for layer, attn_data in enumerate(attn_data_list):
        attn = attn_data['attn'][0]  # assume batch_size=1; output shape = (num_heads, seq_len, seq_len)
        all_attns.append(attn.tolist())
        if include_queries_and_keys:
            queries = attn_data['queries'][0]  # assume batch_size=1; output shape = (num_heads, seq_len, vector_size)
            all_queries.append(queries.tolist())
            keys = attn_data['keys'][0]  # assume batch_size=1; output shape = (num_heads, seq_len, vector_size)
            all_keys.append(keys.tolist())
    results = {
        'attn': all_attns,
        'left_text': tokens,
        'right_text': tokens
    }
    if include_queries_and_keys:
        results.update({
            'queries': all_queries,
            'keys': all_keys,
        })
    return {'all': results}


def get_attention_vilbert(model, tokenizer, batch, include_queries_and_keys=False):

    """Compute representation of the attention for BERT to pass to the d3 visualization

    Args:
      model: BERT model
      tokenizer: BERT tokenizer
      batch
      include_queries_and_keys: Indicates whether to include queries/keys in results

    Returns:
      Dictionary of attn representations with the structure:
      {
        'all': All attention (source = AB, target = AB)
        'aa': Sentence A self-attention (source = A, target = A)
        'bb': Sentence B self-attention (source = B, target = B)
        'ab': Sentence A -> Sentence B attention (source = A, target = B)
        'ba': Sentence B -> Sentence A attention (source = B, target = A)
      }
      where each value is a dictionary:
      {
        'left_text': list of source tokens, to be displayed on the left of the vis
        'right_text': list of target tokens, to be displayed on the right of the vis
        'attn': list of attention matrices, one for each layer. Each has shape [num_heads, source_seq_len, target_seq_len]
        'queries' (optional): list of query vector arrays, one for each layer. Each has shape (num_heads, source_seq_len, vector_size)
        'keys' (optional): list of key vector arrays, one for each layer. Each has shape (num_heads, target_seq_len, vector_size)
      }
    """
    # Call model to get attention data
    model.eval()
    input_ids, input_mask, segment_ids, lm_label_ids, is_next, image_feat, image_loc, image_target, image_label, image_mask = (
        batch
    )

    sents = input_ids.cpu().numpy().tolist()[0]
    sents = tokenizer.convert_ids_to_tokens(sents)
    sents = [sent for sent in sents if sent != "[PAD]"]

    region_num = image_mask[0].sum().item()
    regions = [str(i) for i in range(region_num)]

    masked_loss_t, masked_loss_v, next_sentence_loss, attn_data_list = model(
        input_ids,
        image_feat,
        image_loc,
        segment_ids,
        input_mask,
        image_mask,
        lm_label_ids,
        output_all_attention_masks=True
    )

    attn_data_list_t, attn_data_list_v, attn_data_list_c = attn_data_list

    # Populate map with attn data and, optionally, query, key data
    keys_dict = defaultdict(list)
    queries_dict = defaultdict(list)
    attn_dict = defaultdict(list)

    for layer, attn_data in enumerate(attn_data_list_t):
        # Process attention
        attn = attn_data['attn'][0]  # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        attn_dict['aa'].append(attn[:,:len(sents),:len(sents)].tolist())  # Append A->A attention for layer, across all heads

    for layer, attn_data in enumerate(attn_data_list_v):
        attn = attn_data['attn'][0]  # assume batch_size=1; shape = [num_heads, source_seq_len, target_seq_len]
        attn_dict['bb'].append(attn[:,:region_num,:region_num].tolist())  # Append A->A attention for layer, across all heads

    for layer, attn_data in enumerate(attn_data_list_c):
        attn1 = attn_data['attn1'][0]
        attn2 = attn_data['attn2'][0]

        attn_dict['ab'].append(attn1[:,:len(sents),:region_num].tolist()) 
        attn_dict['ba'].append(attn2[:,:region_num,:len(sents)].tolist())

    results = {
        'aa': {
            'attn': attn_dict['aa'],
            'left_text': sents,
            'right_text': sents
        },
        'bb': {
            'attn': attn_dict['bb'],
            'left_text': regions,
            'right_text': regions
        },
        'ab': {
            'attn': attn_dict['ab'],
            'left_text': sents,
            'right_text': regions
        },
        'ba': {
            'attn': attn_dict['ba'],
            'left_text': regions,
            'right_text': sents
        }
    }

    return results