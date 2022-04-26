import gensim
from gensim.models.fasttext import load_facebook_model
import numpy as np
import torch
from pykeen.datasets import WN18RR


def dict_entry_from_line(line):
    line_list = line.split("\t")
    idx = line_list[0]
    word_list = line_list[1].split("_")
    word = " ".join(word_list[2:-2])
    return idx, word


def get_id_word_dict(dataset_name):

    if dataset_name == "WN18RR":

        id_word_dict = {}
        with open("aux_data/WN18RR/wordnet-mlj12-definitions.txt", "r") as f:
            for line in f:
                sense_id, word = dict_entry_from_line(line)
                id_word_dict[sense_id] = word
        return id_word_dict
    else:
        pass


def get_emb_matrix(init_embeddings_path, dataset_name=None):

    if dataset_name == "WN18RR":
        dataset = WN18RR()
    else:
        raise NotImplementedError("Dataset not implemented")

    entity_dict = dataset.training.entity_to_id

    id_word_dict = get_id_word_dict(dataset_name)

    w2v_model = load_facebook_model(init_embeddings_path)

    emb_matrix = np.zeros((len(entity_dict), 300))

    for entity_id in entity_dict:
        entity_emb_idx = entity_dict[entity_id]
        emb_matrix[entity_emb_idx, :] = w2v_model.wv[id_word_dict[entity_id]]

    return torch.from_numpy(emb_matrix.astype(np.float32))
