from pykeen.pipeline import pipeline
from pykeen.nn.init import PretrainedInitializer
from bert_embs import get_bert_embeddings
import w2v
import json
import argparse
import pathlib
from typing import Union, List
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


def load_configuration(
    path: Union[str, pathlib.Path, os.PathLike]
) -> Mapping[str, Any]:
    """Load a configuration from a JSON or YAML file."""

    path = pathlib.Path(path)
    if path.suffix == ".json":
        with path.open() as file:
            return json.load(file)

    if path.suffix in {".yaml", ".yml"}:
        with path.open() as file:
            return yaml.safe_load(file)

    raise ValueError(
        f"Unknown configuration file format: {path.suffix}. Valid formats: json, yaml"
    )


def get_entity_initializer(
    init: str, embedding_dim, dataset_name, config, vectors_dir="word_vectors", bert_layer=-1, bert_weigh=False, bert_desc=False, bert_layer_weights=None
):
    """Get an Entity embeddings initializer."""

    if init == "fasttext":
        entity_initializer = PretrainedInitializer(
            w2v.get_emb_matrix(
                f"{vectors_dir}/cc.en.{embedding_dim}.bin",
                embedding_dim,
                sub_word=True,
                dataset_name=dataset_name,
            )
        )
    elif init == "w2v":
        entity_initializer = PretrainedInitializer(
            w2v.get_emb_matrix(
                f"{vectors_dir}/word2vec-google-news-{embedding_dim}.bin",
                embedding_dim,
                sub_word=False,
                dataset_name=dataset_name,
            )
        )
    elif init == "glove":
        entity_initializer = PretrainedInitializer(
            w2v.get_emb_matrix(
                f"{vectors_dir}/glove-wiki-gigaword-{embedding_dim}.bin",
                embedding_dim,
                sub_word=False,
                dataset_name=dataset_name,
            )
        )
    elif init == "bert":
        if bert_layer is not None:
            bert_layer = [int(x) for x in bert_layer]
        if bert_layer_weights is not None:
            bert_layer_weights = [float(x) for x in bert_layer_weights]

            assert(len(bert_layer) > 1)
            assert(len(bert_layer) == len(bert_layer_weights))

        bert_emb_matrix = get_bert_embeddings(
            layers=bert_layer,
            layer_weights=bert_layer_weights,
            dataset_name=dataset_name,
            bert_model="prajjwal1/bert-mini",
            use_entity_descriptions=bert_desc,
            weigh_mean=bert_weigh,
        )

        entity_initializer = PretrainedInitializer(bert_emb_matrix)
    else:
        entity_initializer = config["pipeline"]["model_kwargs"]["entity_initializer"]
    return entity_initializer


def pipeline_from_config(
    dataset_name: str,
    model_name: str,
    init: str,
    embdim: int,
    epochs: int,
    vectors_dir: str,
    random_seed: int,
    wandb_group: str,
    bert_layer: List[int],
    bert_layer_weights: List[float],
    bert_stem: bool,
    bert_desc: bool,
    dropout_0: float,
    dropout_1: float,
    dropout_2: float,
):
    """Initialize pipeline parameters from config file."""

    config_path = f"./config/{model_name.lower()}_{dataset_name.lower()}.json"
    config = load_configuration(config_path)

    if embdim is None:
        embedding_dim = config["pipeline"]["model_kwargs"]["embedding_dim"]
    else:
        embedding_dim = embdim

    if epochs is None:
        num_epochs = config["pipeline"]["training_kwargs"]["num_epochs"]
    else:
        num_epochs = epochs

    if dropout_0:
        config["pipeline"]["model_kwargs"]["dropout_0"] = dropout_0

    if dropout_1:
        config["pipeline"]["model_kwargs"]["dropout_1"] = dropout_1

    if dropout_2:
        config["pipeline"]["model_kwargs"]["dropout_2"] = dropout_2

    entity_initializer = get_entity_initializer(
        init, embedding_dim, dataset_name, config, vectors_dir, bert_layer, bert_stem, bert_desc, bert_layer_weights,
    )

    if random_seed is not None:
        config["pipeline"]["random_seed"] = random_seed

    config["pipeline"]["model_kwargs"]["entity_initializer"] = entity_initializer
    config["pipeline"]["model_kwargs"]["embedding_dim"] = embedding_dim
    config["pipeline"]["training_kwargs"]["num_epochs"] = num_epochs

    run_name = f"{init}_{embedding_dim}_{model_name}_{dataset_name}"

    pipeline_kwargs = config["pipeline"]
    pipeline_result = pipeline(
        metadata=dict(
            title=run_name,
        ),
        result_tracker="wandb",
        result_tracker_kwargs=dict(
            project="W2V_for_KGs", entity="eth_ai_center_kg_project", group=wandb_group
        ),
        **pipeline_kwargs,
    )

    pipeline_result.save_to_directory(f"results/{run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        nargs="?",
        help="Which dataset to use: fb15k237 or wn18rr.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tucker",
        nargs="?",
        help="Which model to train: tucker",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="baseline",
        nargs="?",
        help="How to initialise embeddings: baseline, w2v, fasttext, glove, bert",
    )
    parser.add_argument(
        "--embdim",
        type=int,
        default=None,
        nargs="?",
        help="Dimension of embedding vectors, if None then embedding_dim in the .json file will be used",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        nargs="?",
        help="Number of epochs to train the model for",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="word_vectors",
        nargs="?",
        help="Directory where vectors are stored",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        nargs="?",
        help="Random seed for the pipeline",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        nargs="?",
        help="Group name for wandb runs",
    )
    parser.add_argument(
        "--bert_layer",
        default=None,
        nargs="+",
        help="BERT layer to take embeddings from",
    )
    parser.add_argument(
        "--bert_layer_weights",
        default=None,
        nargs="+",
        help="weights for every bert_layer to computer weighted average",
    )
    parser.add_argument(
        "--bert_stem_weighted",
        action="store_true",
        help="weight BERT tokens using stemming",
    )
    parser.add_argument(
        "--bert_desc",
        action="store_true",
        help="use entity descriptions to init BERT embs",
    )
    parser.add_argument(
        "--dropout_0",
        type=float,
        default=None,
        nargs="?",
        help="Dropout rate on TuckER core tensor",
    )
    parser.add_argument(
        "--dropout_1",
        type=float,
        default=None,
        nargs="?",
        help="Dropout rate on ...",
    )
    parser.add_argument(
        "--dropout_2",
        type=float,
        default=None,
        nargs="?",
        help="Dropout rate on ...",
    )

    args = parser.parse_args()

    pipeline_from_config(
        args.dataset,
        args.model,
        args.init,
        args.embdim,
        args.epochs,
        args.vectors_dir,
        args.random_seed,
        args.wandb_group,
        args.bert_layer,
        args.bert_layer_weights,
        args.bert_stem_weighted,
        args.bert_desc,
        args.dropout_0,
        args.dropout_1,
        args.dropout_2,
    )
