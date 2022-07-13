from mimetypes import init
import numpy as np
import torch
import time
from collections import defaultdict
from load_data import Data
from model import *
from rsgd import *
import argparse
import sys
import wandb

sys.path.insert(0, "..")
from w2v import get_emb_matrix, get_emb_matrix_relation
from bert_embs import get_bert_embeddings, get_bert_embeddings_relation


class Experiment:
    def __init__(
        self,
        learning_rate=50,
        dim=40,
        nneg=50,
        model="poincare",
        num_iterations=500,
        batch_size=128,
        cuda=False,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda

    def get_data_idxs(self, data):
        data_idxs = [
            (
                self.entity_idxs[data[i][0]],
                self.relation_idxs[data[i][1]],
                self.entity_idxs[data[i][2]],
            )
            for i in range(len(data))
        ]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        for i in range(0, len(test_data_idxs)):
            data_point = test_data_idxs[i]
            e1_idx = torch.tensor(data_point[0]).to(device)
            r_idx = torch.tensor(data_point[1]).to(device)
            e2_idx = torch.tensor(data_point[2]).to(device)
            predictions_s = model.forward(
                e1_idx.repeat(len(d.entities)),
                r_idx.repeat(len(d.entities)),
                range(len(d.entities)),
            )

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions_s[e2_idx].item()
            predictions_s[filt] = -np.Inf
            predictions_s[e1_idx] = -np.Inf
            predictions_s[e2_idx] = target_value

            sort_values, sort_idxs = torch.sort(predictions_s, descending=True)

            sort_idxs = sort_idxs.cpu().numpy()
            rank = np.where(sort_idxs == e2_idx.item())[0][0]
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

        print("Hits @10: {0}".format(np.mean(hits[9])))
        print("Hits @3: {0}".format(np.mean(hits[2])))
        print("Hits @1: {0}".format(np.mean(hits[0])))
        print("Mean rank: {0}".format(np.mean(ranks)))
        print("Mean reciprocal rank: {0}".format(np.mean(1.0 / np.array(ranks))))
        wandb.log({
        'Hits @10': np.mean(hits[9]),
        'Hits @3': np.mean(hits[2]),
        'Hits @1': np.mean(hits[0]),
        'Mean rank': np.mean(ranks),
        'Mean reciprocal rank': np.mean(1./np.array(ranks))
        })
        
        
    def train_and_eval(self):
        print("Training the %s model..." % self.model)
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        if args.init == "glove":
            entity_matrix, entity_dict = get_emb_matrix(
                init_embeddings_path=f"../{args.vectors_dir}/glove-wiki-gigaword-200.bin",
                embedding_dim=200,
                dataset_name=dataset,
                mure_init=True,
            )

            new_entity_matrix = torch.zeros(len(d.entities), 200)
            for entity_id in entity_dict:
                entity_vector = entity_matrix[entity_dict[entity_id], :]
                new_entity_matrix[self.entity_idxs[entity_id], :] = entity_vector
            entity_matrix = new_entity_matrix
        elif args.init == "bert":
            entity_matrix, entity_dict = get_bert_embeddings(
                layers=args.bert_layer,
                layer_weights=args.bert_layer_weights,
                dataset_name=dataset,
                bert_model="prajjwal1/bert-mini",
                use_entity_descriptions=args.bert_desc,
                weigh_mean=False,
                mure_init=True,
            )
            new_entity_matrix = torch.zeros(len(d.entities), 256)
            for entity_id in entity_dict:
                entity_vector = entity_matrix[entity_dict[entity_id], :]
                new_entity_matrix[self.entity_idxs[entity_id], :] = entity_vector
            entity_matrix = new_entity_matrix
        else:
            entity_matrix = None
        if args.relation_init == "glove":
            rel_vec = get_emb_matrix_relation(
                init_embeddings_path=f"../{args.vectors_dir}/glove-wiki-gigaword-200.bin",
                embedding_dim=200,
                dataset_name=dataset,
                sub_word=False,
            )
        elif args.relation_init == "bert":
            rel_vec = get_bert_embeddings_relation(
                layers=args.bert_layer,
                dataset_name=dataset,
                bert_model="prajjwal1/bert-mini",
            )
        else:
            rel_vec = None
        if args.relation_matrix_init == "glove":
            rel_mat = get_emb_matrix_relation(
                init_embeddings_path=f"../{args.vectors_dir}/glove-wiki-gigaword-200.bin",
                embedding_dim=200,
                dataset_name=dataset,
                sub_word=False,
            )
        elif args.relation_matrix_init == "bert":
            rel_mat = get_bert_embeddings_relation(
                layers=args.bert_layer,
                dataset_name=dataset,
                bert_model="prajjwal1/bert-mini",
            )
        else:
            rel_mat = None

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model == "poincare":
            model = MuRP(d, self.dim)
        else:
            model = MuRE(
                d, self.dim, entity_mat=entity_matrix, rel_vec=rel_vec, rel_mat=rel_mat
            )
        param_names = [name for name, _ in model.named_parameters()]
        opt = RiemannianSGD(
            model.parameters(), lr=self.learning_rate, param_names=param_names
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        er_vocab = self.get_er_vocab(train_data_idxs)

        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()

            losses = []
            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j : j + self.batch_size])
                negsamples = np.random.choice(
                    list(self.entity_idxs.values()),
                    size=(data_batch.shape[0], self.nneg),
                )

                e1_idx = torch.tensor(
                    np.tile(
                        np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1] + 1)
                    )
                )
                r_idx = torch.tensor(
                    np.tile(
                        np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1] + 1)
                    )
                )
                e2_idx = torch.tensor(
                    np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1)
                )

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)
                targets.to(device)

                opt.zero_grad()

                e1_idx = e1_idx.to(device)
                r_idx = r_idx.to(device)
                e2_idx = e2_idx.to(device)
                targets = targets.to(device)

                predictions = model.forward(e1_idx, r_idx, e2_idx)
                loss = model.loss(predictions.cpu(), targets.cpu())
                loss.backward()
                opt.step()
                losses.append(loss.item())
            print(it)
            print(time.time() - start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it % 5:
                    print("Test:")
                    self.evaluate(model, d.test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="wn18rr",
        nargs="?",
        help="Which dataset to use: fb15k-237 or wn18rr.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="euclidian",
        nargs="?",
        help="Which model to use: poincare or euclidean.",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500,
        nargs="?",
        help="Number of iterations.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, nargs="?", help="Batch size."
    )
    parser.add_argument(
        "--nneg", type=int, default=50, nargs="?", help="Number of negative samples."
    )
    parser.add_argument(
        "--cuda",
        type=bool,
        default=False,
        nargs="?",
        help="Whether to use cuda (GPU) or not (CPU).",
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        help="How to initialise entity embeddings: baseline, glove, bert",
    )
    parser.add_argument(
        "--relation_init",
        type=str,
        default=None,
        help="How to initialise relation embeddings: baseline, glove, bert",
    )
    parser.add_argument(
        "--relation_matrix_init",
        type=str,
        default=None,
        nargs="?",
        help="How to initialise relation matrix mure: baseline, glove, bert",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=40,
        nargs="?",
        help="Random seed for the pipeline",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default="word_vectors",
        nargs="?",
        help="Directory where vectors are stored",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        nargs="?",
        help="training learning rate",  
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
        default=[0],
        nargs="+",
        help="BERT layer to take embeddings from",
    )
    parser.add_argument(
        "--bert_layer_weight_1",
        default=None,
        type=float,
        nargs="?",
        help="weights for first bert_layer to computer weighted average",
    )
    parser.add_argument(
        "--bert_layer_weight_2",
        default=None,
        type=float,
        nargs="?",
        help="weights for second bert_layer to computer weighted average",
    )
    parser.add_argument(
        "--bert_layer_weight_3",
        default=None,
        type=float,
        nargs="?",
        help="weights for third bert_layer to computer weighted average",
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

    args = parser.parse_args()

    bert_layer_weights = []
    if args.bert_layer_weight_1 is not None:
        bert_layer_weights.append(args.bert_layer_weight_1)
    if args.bert_layer_weight_2 is not None:
        bert_layer_weights.append(args.bert_layer_weight_2)
    if args.bert_layer_weight_3 is not None:
        bert_layer_weights.append(args.bert_layer_weight_3)

    if args.bert_layer is not None:
        args.bert_layer = [int(x) for x in args.bert_layer]
    if len(bert_layer_weights) > 0:
        bert_layer_weights = [float(x) for x in bert_layer_weights]

        assert len(args.bert_layer) > 1
        assert len(args.bert_layer) == len(bert_layer_weights)

        args.bert_layer_weights = bert_layer_weights
    else:
        args.bert_layer_weights = None

    dataset = args.dataset
    dataset = dataset.lower()
    if args.lr is None:
        args.lr = 50 if dataset == "wn18rr" else 10
    args.dim = 256 if args.init == "bert" else 200
    
    run_name = f"{args.init}_{args.dim}_OgMuRE_{dataset}"
    wandb.init(name=run_name, entity="eth_ai_center_kg_project", project="W2V_for_KGs", group=args.wandb_group)
    wandb.config.use_pykeen = False
    wandb.config.update(args)

    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)
    experiment = Experiment(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        dim=args.dim,
        cuda=args.cuda,
        nneg=args.nneg,
        model=args.model,
    )
    experiment.train_and_eval()
