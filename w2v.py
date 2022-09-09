import gensim
from gensim.models.fasttext import load_facebook_model
import numpy as np
import torch
from pykeen.datasets import WN18RR, FB15k237
import json
import pdb
from tqdm import tqdm

def get_id_description_dict(dataset_name):
    if dataset_name == "wn18rr":
        id_desc_dict = {}

        try:
            with open("aux_data/WN18RR/wordnet-mlj12-definitions.txt", "r") as f:
                for line in f:
                    line_list = line.split("\t")
                    idx = line_list[0]
                    desc = line_list[-1]

                    id_desc_dict[idx] = desc
        except FileNotFoundError:
            with open("../aux_data/WN18RR/wordnet-mlj12-definitions.txt", "r") as f:
                for line in f:
                    line_list = line.split("\t")
                    idx = line_list[0]
                    desc = line_list[-1]

                    id_desc_dict[idx] = desc

        return id_desc_dict
    else:
        raise ValueError


def dict_entry_from_line(line, sub_word=True):
    line_list = line.split("\t")
    idx = line_list[0]
    word_list = line_list[1].split("_")
    if sub_word:
        word = " ".join(word_list[2:-2])
    else:
        word = word_list[2:-2]
    return idx, word


def proc_fb_label(label, sub_word=True):
    if sub_word:
        return label
    else:
        label = label.replace(",", "")
        label_list = label.split(" ")
        return label_list


def get_id_word_dict(dataset_name, sub_word=True, mure_init=False):
    if dataset_name == "wn18rr":
        id_word_dict = {}
        path = "aux_data/WN18RR/wordnet-mlj12-definitions.txt"
        if mure_init:
            path = "../" + path
        with open(path, "r") as f:
            for line in f:
                sense_id, word = dict_entry_from_line(line, sub_word)
                id_word_dict[sense_id] = word
    elif dataset_name == "fb15k237":
        path = "aux_data/entity2wikidata.json"
        if mure_init:
            path = "../" + path
        with open(path, "r") as f:
            entity2wikidata = json.load(f)
        id_word_dict = {
            entity_id: proc_fb_label(entity2wikidata[entity_id]["label"])
            for entity_id in entity2wikidata
        }
    else:
        raise NotImplementedError
    return id_word_dict


def get_vector_from_word_list(word_list, word_vectors, embedding_dim, rel_fb_mod=False, prev_vec=None):
    vector = np.zeros(embedding_dim)
    
    if rel_fb_mod:
        n_words = 0
        for word in word_list:
            try:
                vector += word_vectors[word]
                n_words += 1
            except:
                pass
        if n_words > 0:
            return vector/n_words
        else:
            return prev_vec
    else:
        for word in word_list:
            try:
                vector += word_vectors[word]
            except:
                tmp = torch.empty(embedding_dim)
                torch.nn.init.normal_(tmp)
                vector += tmp.numpy()
        return vector / len(word_list)


def get_emb_matrix(
    init_embeddings_path, embedding_dim, sub_word=True, dataset_name=None, mure_init=False
):
    if dataset_name == "wn18rr":
        dataset = WN18RR()
    elif dataset_name == "fb15k237":
        dataset = FB15k237()
    else:
        raise NotImplementedError

    entity_dict = dataset.training.entity_to_id

    id_word_dict = get_id_word_dict(dataset_name, sub_word, mure_init=mure_init)

    if "cc" in init_embeddings_path:
        w2v_model = load_facebook_model(init_embeddings_path).wv
    elif "glove" in init_embeddings_path:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            init_embeddings_path, binary=True
        )
    elif "word2vec" in init_embeddings_path:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            init_embeddings_path, binary=True
        )

    emb_matrix = np.zeros((len(entity_dict), embedding_dim))

    for entity_id in entity_dict:
        entity_emb_idx = entity_dict[entity_id]
        if sub_word:
            try:
                emb_matrix[entity_emb_idx, :] = w2v_model[id_word_dict[entity_id]]
            except:
                tmp = torch.empty(embedding_dim)
                torch.nn.init.normal_(tmp)
                emb_matrix[entity_emb_idx, :] = tmp.numpy()
        else:
            try:
                emb_matrix[entity_emb_idx, :] = get_vector_from_word_list(
                    id_word_dict[entity_id], w2v_model, embedding_dim
                )
            except:
                tmp = torch.empty(embedding_dim)
                torch.nn.init.normal_(tmp)
                emb_matrix[entity_emb_idx, :] = tmp.numpy()

    if mure_init:
        return torch.from_numpy(emb_matrix.astype(np.float32)), entity_dict
    else:
        return torch.from_numpy(emb_matrix.astype(np.float32))


def get_emb_matrix_relation(
    init_embeddings_path, embedding_dim, sub_word=True, dataset_name=None
):

    if dataset_name == "wn18rr":
        dataset = WN18RR()
    elif dataset_name == "fb15k237":
        dataset = FB15k237()
    else:
        raise NotImplementedError

    relation_dict = dataset.training.relation_to_id

    if "cc" in init_embeddings_path:
        w2v_model = load_facebook_model(init_embeddings_path).wv
    elif "glove" in init_embeddings_path:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            init_embeddings_path, binary=True
        )
    elif "word2vec" in init_embeddings_path:
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            init_embeddings_path, binary=True
        )

    emb_matrix = np.zeros((len(relation_dict), embedding_dim))

    if dataset_name == "wn18rr":
        for relation in relation_dict:
            relation_id = relation_dict[relation]
            word_list = relation.split("_")
            word_list = [i for i in word_list if i]
            if sub_word:
                raise NotImplementedError  # TODO (if only there were 48hrs in a day)
            else:
                try:
                    emb_matrix[relation_id, :] = get_vector_from_word_list(
                        word_list, w2v_model, embedding_dim
                    )
                except:
                    tmp = torch.empty(embedding_dim)
                    torch.nn.init.normal_(tmp)
                    emb_matrix[relation_id, :] = tmp.numpy()
                    print("exception")

    elif dataset_name == "fb15k237":
        for relation in tqdm(relation_dict):
            relation_id = relation_dict[relation]
            path_components = relation[1:].split("/")
            path_components.reverse()
            emb = torch.zeros(embedding_dim)
            sum_of_weights = 0
            for i in range(len(path_components)):
                sum_of_weights += 2 ** (-(i + 1))
            for i, component in enumerate(path_components):
                component = component.replace(".", "")
                word_list = relation.split("_")
                word_list = [i for i in word_list if i]
                if sub_word:
                    raise NotImplementedError  # TODO (if only there were 48hrs in a day)
                else:
                    comp_weight = (2 ** (-(i + 1)))/sum_of_weights
                    emb += comp_weight * get_vector_from_word_list(
                        word_list, w2v_model, embedding_dim, rel_fb_mod=True, prev_vec=emb
                    )
            if torch.count_nonzero(emb) < 1:
                emb = torch.empty(embedding_dim)
                torch.nn.init.normal_(emb)

            emb_matrix[relation_id, :] = emb

    return torch.from_numpy(emb_matrix.astype(np.float32))
