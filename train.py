#TODO ReadMe with env variables
#TODO Setup ReadME

from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer
import w2v
import json
import argparse
import pathlib
import yaml
from typing import (
    Any,
    Mapping,
    Union,
)
import os

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
    else:
        entity_initializer = "xavier_normal"
    return entity_initializer

def pipeline_from_config(dataset_name: str,model_name :str,init: str):
    """Initialize pipeline parameters from config file."""

    config_path = f"./config/{model_name.lower()}_{dataset_name.lower()}.json"
    config = load_configuration(config_path)
    embedding_dim = config["pipeline"]["model_kwargs"]["embedding_dim"]
    entity_initializer = get_entity_initializer(init, embedding_dim, dataset_name)
    config["pipeline"]["model_kwargs"]["entity_initializer"] = entity_initializer

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
    parser.add_argument("--init", type=str, default="w2v", nargs="?",
                    help="How to initialise embeddings: baseline, w2v, glove")

    args = parser.parse_args()
    pipeline_from_config(args.dataset,args.model,args.init)