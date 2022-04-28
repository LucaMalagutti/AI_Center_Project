import gensim
from gensim.models.fasttext import load_facebook_model
import numpy as np
import torch
from pykeen.datasets import WN18RR


def dict_entry_from_line(line, sub_word=True):
    line_list = line.split("\t")
    idx = line_list[0]
    word_list = line_list[1].split("_")
    if sub_word:
        word = " ".join(word_list[2:-2])
    else:
        word = word_list[2:-2]
    return idx, word

def get_vector_from_word_list(word_list, word_vectors, embeddings_dim):
    vector = np.zeros(embeddings_dim)
    for word in word_list:
        try:
            vector += word_vectors[word]
        except:
            tmp = torch.empty(200)
            torch.nn.init.normal_(tmp)
            vector += tmp.numpy()
    return vector/len(word_list)


def get_id_word_dict(dataset_name, sub_word=True):

    if dataset_name == "WN18RR":
        id_word_dict = {}
        with open("aux_data/WN18RR/wordnet-mlj12-definitions.txt", "r") as f:
            for line in f:
                sense_id, word = dict_entry_from_line(line, sub_word)
                id_word_dict[sense_id] = word
        return id_word_dict
    else:
        pass


def get_emb_matrix(init_embeddings_path, embeddings_dim, sub_word=True, dataset_name=None):

    if dataset_name == "WN18RR":
        dataset = WN18RR()
    else:
        raise NotImplementedError("Dataset not implemented")

    entity_dict = dataset.training.entity_to_id

    id_word_dict = get_id_word_dict(dataset_name, sub_word)

    if "cc.en.200.bin" in init_embeddings_path:
        w2v_model = load_facebook_model(init_embeddings_path).wv
    elif "glove-wiki-gigaword-200.bin" in init_embeddings_path:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(init_embeddings_path, binary=True)

    emb_matrix = np.zeros((len(entity_dict), embeddings_dim))

    for entity_id in entity_dict:
        entity_emb_idx = entity_dict[entity_id]
        if sub_word:
            emb_matrix[entity_emb_idx, :] = w2v_model[id_word_dict[entity_id]]
        else:
            emb_matrix[entity_emb_idx, :] = get_vector_from_word_list(id_word_dict[entity_id], w2v_model, embeddings_dim)

    return torch.from_numpy(emb_matrix.astype(np.float32))