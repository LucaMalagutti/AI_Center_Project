import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, PretrainedConfig
from w2v import get_id_word_dict, get_id_description_dict
import pickle
from tqdm import tqdm
from pykeen.datasets import WN18RR, FB15k237
from nltk import PorterStemmer
import pylcs
import pdb

def get_stemmer_mean_weights(stemmer, tokenizer, word):
    stemmed = stemmer.stem(word)
    encoded_ids = tokenizer.encode_plus(word).input_ids
    num_tokens = len(encoded_ids) - 2

    weights = [1.0]

    if num_tokens > 1 and word is not None:
        decoded_tokens = [tokenizer.decode(encoded_ids[i]).replace("##", "") for i in range(1, len(encoded_ids)-1)]

        begin = 0
        end = 1
        max_lcs = -1

        best_end = -1
        best_begin = -1

        while begin < len(decoded_tokens):
            lcs = pylcs.lcs(stemmed, "".join(decoded_tokens[begin:end]))
            if lcs > max_lcs:
                max_lcs = lcs
                best_end = end
                best_begin = begin
            
                if end < len(decoded_tokens):
                    end += 1
                else:
                    begin += 1 

            else:
                begin += 1
                if end < len(decoded_tokens) and begin + 1 >= end:
                    end += 1

        weights = [1/(best_end-best_begin) for _ in range(num_tokens)]
        for i in range(best_begin, best_end):
            weights[i] = 1
    
    return weights


def get_hidden_states(encoded, token_ids_word, model, tokenizer, layers, weigh_mean=False, entity_name=None, layer_weights=None):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
    Select only those subword token outputs that belong to our word of interest
    and average them."""
    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states

    # weight average over BERT layers if more than one is included
    if layer_weights is not None:
        output = torch.Tensor(np.average([states[i] for i in layers], axis=0, weights=layer_weights)).squeeze()
    else:
        output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

    # Only select the tokens that constitute the requested word
    word_tokens_output = output[token_ids_word]

    if weigh_mean:
        stemmer = PorterStemmer()
        weights = []
        for word in entity_name.split(" "):
            weights.extend(get_stemmer_mean_weights(stemmer, tokenizer, word))

        if len(weights) == len(word_tokens_output):
            return torch.Tensor(np.average(word_tokens_output, axis=0, weights=weights))

    return word_tokens_output.mean(dim=0)


def get_word_idx(sent: str, word: str):
    return sent.split(" ").index(word)


def get_word_vector(sent, idx_list, tokenizer, model, layers, weigh_mean=False, entity_name=None, layer_weights=None):
    """Get a word vector by first tokenizing the input sentence, getting all token idxs
    that make up the word of interest, and then `get_hidden_states`."""
    encoded = tokenizer.encode_plus(sent, return_tensors="pt")

    token_ids_words = np.array([w_id in idx_list for w_id in encoded.word_ids()])

    return get_hidden_states(encoded, token_ids_words, model, tokenizer, layers, weigh_mean, entity_name, layer_weights)


def get_bert_embeddings(layers=[-1], dataset_name="wn18rr", bert_model="prajjwal1/bert-mini", use_entity_descriptions=False, weigh_mean=False, layer_weights=None):
    dataset_name = dataset_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model, output_hidden_states=True)

    model_config = PretrainedConfig.from_pretrained(bert_model)
    embedding_dim = model_config.hidden_size

    if dataset_name == "wn18rr":
        dataset = WN18RR()
    elif dataset_name == "fb15k237":
        dataset = FB15k237()
    else:
        raise NotImplementedError("Dataset not implemented")

    entity_dict = dataset.training.entity_to_id

    entity_id_to_word = get_id_word_dict(dataset_name, sub_word=True)
    if use_entity_descriptions:
        if dataset_name == "wn18rr":
            entity_id_to_description = get_id_description_dict(dataset_name)
        else:
            raise ValueError("No descriptions available for FB15k237 dataset")

    emb_matrix = np.zeros((len(entity_dict), embedding_dim))

    for entity_id in tqdm(entity_dict):
        entity_emb_idx = entity_dict[entity_id]
        try:
            entity_name = entity_id_to_word[str(entity_id)]

            if use_entity_descriptions:
                input_sentence = (
                    f"{entity_name} : {entity_id_to_description[str(entity_id)]}"
                )
            else:
                input_sentence = entity_name

            idx_list = [get_word_idx(input_sentence, word) for word in entity_name.split(" ")]
            emb = get_word_vector(input_sentence, idx_list, tokenizer, model, layers, weigh_mean, entity_name, layer_weights)

            emb_matrix[entity_emb_idx, :] = emb
        except KeyError as _:
            # print(f"{entity_id} missing")
            pass

    emb_matrix = torch.from_numpy(emb_matrix.astype(np.float32))

    return emb_matrix



def get_bert_embeddings_relation(layers=[-1], dataset_name="WN18RR", bert_model="prajjwal1/bert-mini"):
    dataset_name = dataset_name.lower()
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model, output_hidden_states=True)

    model_config = PretrainedConfig.from_pretrained(bert_model)
    embedding_dim = model_config.hidden_size

    if dataset_name == "wn18rr":
        dataset = WN18RR()
    else:
        raise NotImplementedError("Dataset not implemented")

    relation_dict = dataset.training.relation_to_id
    
    emb_matrix = np.zeros((len(relation_dict), embedding_dim))

    for relation in tqdm(relation_dict):
        relation_id = relation_dict[relation]
        input_sentence = relation[1:].replace('_', ' ')
        idx_list = [get_word_idx(input_sentence, word) for word in input_sentence.split(" ")]
        emb = get_word_vector(input_sentence, idx_list, tokenizer, model, layers, entity_name=input_sentence)
        emb_matrix[relation_id, :] = emb

    emb_matrix = torch.from_numpy(emb_matrix.astype(np.float32))
    return emb_matrix

if __name__ == '__main__':
    OUTPUT_NAME = "bert-mini_embs.pickle"
    emb_matrix = get_bert_embeddings()

    with open(f'word_vectors/{OUTPUT_NAME}', 'wb') as handle:
        pickle.dump(emb_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
