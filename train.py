from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer
import w2v
import json
import argparse
import pathlib
import pickle
import torch
import yaml
from typing import (
    Any,
    Mapping,
    Union,
)
import os
import dotenv

# Load environment variables from `.env`.
dotenv.load_dotenv(override=True)

def load_configuration(path: Union[str, pathlib.Path, os.PathLike]) -> Mapping[str, Any]:
    """Load a configuration from a JSON or YAML file."""
    
    path = pathlib.Path(path)
    if path.suffix == ".json":
        with path.open() as file:
            return json.load(file)

    if path.suffix in {".yaml", ".yml"}:
        with path.open() as file:
            return yaml.safe_load(file)

    raise ValueError(f"Unknown configuration file format: {path.suffix}. Valid formats: json, yaml")

def get_entity_initializer(init:str, embedding_dim, dataset_name):
    """Get an Entity embeddings initializer."""

    if init == "w2v":
        entity_initializer = PretrainedInitializer(w2v.get_emb_matrix(f"word_vectors/cc.en.{embedding_dim}.bin", embedding_dim, sub_word=True, dataset_name=dataset_name.upper()))
    elif init == "glove":
        entity_initializer = PretrainedInitializer(w2v.get_emb_matrix(f"word_vectors/glove-wiki-gigaword-{embedding_dim}.bin", embedding_dim, sub_word=False, dataset_name=dataset_name.upper()))
    elif init == "bert":
        with open('word_vectors/bert-mini_no_def.pickle', 'rb') as f:
            bert_emb_matrix = pickle.load(f)
        print(bert_emb_matrix.shape) # prints torch.Size([40559, 256])
        
        entity_initializer = PretrainedInitializer(bert_emb_matrix)
    else:
        entity_initializer = "xavier_normal"
    return entity_initializer

def pipeline_from_config(dataset_name: str,model_name :str,init: str, embdim : int):
    """Initialize pipeline parameters from config file."""

    config_path = f"./config/{model_name.lower()}_{dataset_name.lower()}.json"
    config = load_configuration(config_path)

    if embdim is None:
        embedding_dim = config["pipeline"]["model_kwargs"]["embedding_dim"]
    else:
        embedding_dim = embdim

    entity_initializer = get_entity_initializer(init, embedding_dim, dataset_name)

    config["pipeline"]["model_kwargs"]["entity_initializer"] = entity_initializer
    config["pipeline"]["model_kwargs"]["embedding_dim"] = embedding_dim

    run_name = f"{init}_{embedding_dim}_{model_name}_{dataset_name}"

    pipeline_kwargs = config["pipeline"]
    pipeline_result = pipeline(
        metadata= dict(
        title = run_name,
        ),
        result_tracker='wandb',
        result_tracker_kwargs=dict(
            project='W2V_for_KGs',
            entity = 'eth_ai_center_kg_project',
        ),
        **pipeline_kwargs
    )


    pipeline_result.save_to_directory(f'results/{run_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wn18rr", nargs="?",
                    help="Which dataset to use: fb15k237 or wn18rr.")
    parser.add_argument("--model", type=str, default="tucker", nargs="?",
                    help="Whihc model to train: tucker")
    parser.add_argument("--init", type=str, default="baseline", nargs="?",
                    help="How to initialise embeddings: baseline, w2v, glove, bert")
    parser.add_argument("--embdim", type=int, default=None, nargs="?",
                    help="Dimension of embedding vectors, if None then embedding_dim in the .json file will be used")

    args = parser.parse_args()
    pipeline_from_config(args.dataset,args.model,args.init,args.embdim)