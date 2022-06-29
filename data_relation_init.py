from collections import defaultdict
from re import I
from pykeen import datasets
from w2v import get_id_word_dict
from bert_embs import get_word_idx, get_word_vector
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import pickle
import torch
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_bert_entity_vector(entity_name, bert_tokenizer, bert_model):
    return get_word_vector(
        sent=entity_name,
        idx_list=[get_word_idx(entity_name, word) for word in entity_name.split(" ")],
        tokenizer = bert_tokenizer,
        model = bert_model,
        layers=[0]
    )


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def construct_data_relation_matrix(model: str, dataset_name: str, we_init: str, absolute_coeffs=True):
    bert_tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-mini")
    bert_model = AutoModel.from_pretrained("prajjwal1/bert-mini", output_hidden_states=True)
    emb_dim = 256
    
    if dataset_name.lower() == "wn18rr":
        dataset = datasets.WN18RR()
        id_to_entity_name_map = get_id_word_dict("wn18rr")
    elif dataset_name.lower() == "fb15k237":
        dataset = datasets.FB15k237()
        id_to_entity_name_map = get_id_word_dict("fb15k237")

    entities_per_relation = defaultdict(list)
    key_error_counter = 0
    for triple in dataset.training.triples:
        try:
            entities_per_relation[triple[1]].append((id_to_entity_name_map[triple[0]], id_to_entity_name_map[triple[2]]))
        except KeyError:
            key_error_counter += 1
    
    if dataset_name.lower() == "fb15k237":
        print(f"{key_error_counter} triple(s) discarded due to FB entity data missing")
    
    print("Initializing entity vectors...")
    entity_vectors = dict()
    for _, rel_triples in entities_per_relation.items():
        for triple in tqdm(rel_triples):
            if triple[0] not in entity_vectors:
                entity_vectors[triple[0]] = get_bert_entity_vector(triple[0], bert_tokenizer, bert_model)
            if triple[1] not in entity_vectors:
                entity_vectors[triple[1]] = get_bert_entity_vector(triple[1], bert_tokenizer, bert_model)

    print("Initializing relation matrix")
    relation_vectors = dict()
    for rel_name in entities_per_relation:
        rel_vector = np.zeros((emb_dim, 1))

        for dim_idx in tqdm(range(emb_dim)):
            dim_counter = 0
            for rel_triple in entities_per_relation[rel_name]:
                if model.lower() == "distmult":
                    if entity_vectors[rel_triple[0]][dim_idx] * entity_vectors[rel_triple[1]][dim_idx] > 0:
                        dim_counter += 1
            
            if absolute_coeffs:
                if dim_counter > len(entities_per_relation[rel_name])/2:
                    rel_vector[dim_idx] = 1.0
                else:
                    rel_vector[dim_idx] = -1.0
            else:
                if len(entities_per_relation[rel_name]):
                    rel_vector[dim_idx] = dim_counter / len(entities_per_relation[rel_name])
                else:
                    rel_vector[dim_idx] = random.random()
        
        relation_vectors[rel_name] = rel_vector

    if not absolute_coeffs:
        relation_vectors = {k: normalize_vector(v) for k, v in relation_vectors.items()}

    relation_to_id_map = dataset.training.relation_to_id
    relation_matrix = np.zeros((len(entities_per_relation), emb_dim))

    for relation_name, relation_vector in relation_vectors.items():
        relation_matrix[relation_to_id_map[relation_name], :] = relation_vector.squeeze()

    output_folder_rel_path = os.path.join("aux_data", "data_rel_matrices")
    if not os.path.exists(output_folder_rel_path):
        os.makedirs(output_folder_rel_path)

    with open(os.path.join(output_folder_rel_path, f"{model.lower()}_{dataset_name.lower()}_{we_init.lower()}.pkl"), 'wb') as pickle_f:
        pickle.dump(relation_matrix, pickle_f, protocol=pickle.HIGHEST_PROTOCOL)

    return torch.from_numpy(relation_matrix.astype(np.float32))


def get_data_relation_matrix(model: str, dataset: str, we_init: str):
    if model.lower() != "distmult":
        raise NotImplementedError("Only distmult supported so far")
    
    if we_init.lower() != "bert":
        raise NotImplementedError("Only BERT initialization supported so far")
    
    if dataset.lower() not in ["wn18rr", "fb15k237"]:
        raise NotImplementedError(f"Only WN and FB datasets are supported, not {dataset.lower()}")

    saved_matrix_path = os.path.join("aux_data", "data_rel_matrices", f"{model.lower()}_{dataset.lower()}_{we_init.lower()}.pkl")
    if os.path.exists(saved_matrix_path):
        with open(saved_matrix_path, 'rb') as f:
            rel_matrix = pickle.load(f)
        return torch.from_numpy(rel_matrix.astype(np.float32))
    
    else:
        return construct_data_relation_matrix(model, dataset, we_init)

if __name__ == "__main__":
    # script inputs
    model = "distmult"
    dataset = "fb15k237"
    we_init = "bert"

    get_data_relation_matrix(model, dataset, we_init)