import numpy as np
import torch
import time
from collections import defaultdict
from load_data import Data
from model import *
from rsgd import *
from torch.optim import Adam
import argparse
from sklearn.preprocessing import normalize
import sys
import wandb
import os


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
        opt=None,
        num_iterations=500,
        batch_size=128,
        cuda=False,
        transe_arch=False,
        transe_loss=False,
        mult_factor=None,
        transe_enable_bias=False,
        transe_bias_mode=None,
        transe_bias_init=None,
        transe_enable_mtx=False,
        transe_enable_vec=False,
        distmult_score_function=False,
        distmult_sqdist=False,
        distmult_sqdist_mode=None,
        normalize_at_step=False,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.opt = opt
        self.transe_arch = transe_arch
        self.transe_loss = transe_loss
        self.mult_factor = mult_factor
        self.transe_enable_bias = transe_enable_bias
        self.transe_bias_mode = transe_bias_mode
        self.transe_bias_init = transe_bias_init
        self.transe_enable_mtx = transe_enable_mtx
        self.transe_enable_vec = transe_enable_vec
        self.distmult_score_function = distmult_score_function
        self.distmult_sqdist = distmult_sqdist
        self.distmult_sqdist_mode = distmult_sqdist_mode
        self.normalize_at_step = normalize_at_step

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
        wandb.log(
            {
                "both.realistic.hits_at_10": np.mean(hits[9]),
                "both.realistic.hits_at_5": np.mean(hits[4]),
                "both.realistic.hits_at_3": np.mean(hits[2]),
                "both.realistic.hits_at_1": np.mean(hits[0]),
                "both.realistic.arithmetic_mean_rank": np.mean(ranks),
                "both.realistic.adjusted_inverse_harmonic_mean_rank": np.mean(1.0 / np.array(ranks)),
            }
        )

        return np.mean(hits[9])

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

            if args.normalize_entity_mtx:
                entity_matrix = normalize(entity_matrix, axis=1)
                entity_matrix = torch.from_numpy(entity_matrix.astype(np.float32))

            new_entity_matrix = torch.tensor(
                np.random.uniform(-1, 1, (len(d.entities), self.dim))
            )
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
                entity_and_relation_input=args.bert_entity_and_relation,
            )

            if args.normalize_entity_mtx:
                entity_matrix = normalize(entity_matrix, axis=1)
                entity_matrix = torch.from_numpy(entity_matrix.astype(np.float32))

            new_entity_matrix = torch.tensor(
                np.random.uniform(-1, 1, (len(d.entities), self.dim))
            )
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

            if args.normalize_rel_vec:
                rel_vec = normalize(rel_vec, axis=1)
                rel_vec = torch.from_numpy(rel_vec.astype(np.float32))
        elif args.relation_init == "bert":
            rel_vec = get_bert_embeddings_relation(
                layers=args.bert_layer,
                dataset_name=dataset,
                bert_model="prajjwal1/bert-mini",
            )

            if args.normalize_rel_vec:
                rel_vec = normalize(rel_vec, axis=1)
                rel_vec = torch.from_numpy(rel_vec.astype(np.float32))
        else:
            rel_vec = None
        if args.relation_matrix_init == "glove":
            rel_mat = get_emb_matrix_relation(
                init_embeddings_path=f"../{args.vectors_dir}/glove-wiki-gigaword-200.bin",
                embedding_dim=200,
                dataset_name=dataset,
                sub_word=False,
            )

            if args.normalize_rel_mat:
                rel_mat = normalize(rel_mat, axis=1)
                rel_mat = torch.from_numpy(rel_mat.astype(np.float32))
        elif args.relation_matrix_init == "bert":
            rel_mat = get_bert_embeddings_relation(
                layers=args.bert_layer,
                dataset_name=dataset,
                bert_model="prajjwal1/bert-mini",
            )

            if args.normalize_rel_mat:
                rel_mat = normalize(rel_mat, axis=1)
                rel_mat = torch.from_numpy(rel_mat.astype(np.float32))
        else:
            rel_mat = None

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.model == "poincare":
            model = MuRP(d, self.dim, entity_mat=entity_matrix)
        elif self.transe_arch:
            model = MuRE_TransE(
                d,
                self.dim,
                entity_mat=entity_matrix,
                rel_vec=rel_vec,
                transe_loss=self.transe_loss,
                mult_factor=self.mult_factor,
                transe_enable_bias=self.transe_enable_bias,
                transe_bias_mode=self.transe_bias_mode,
                transe_bias_init=self.transe_bias_init,
                transe_enable_mtx=self.transe_enable_mtx,
                transe_enable_vec=self.transe_enable_vec,
                distmult_score_function=self.distmult_score_function,
                distmult_sqdist=self.distmult_sqdist,
                distmult_sqdist_mode=self.distmult_sqdist_mode,
            )
        else:
            model = MuRE(
                d,
                self.dim,
                entity_mat=entity_matrix,
                rel_vec=rel_vec,
                rel_mat=rel_mat,
                mult_factor=self.mult_factor,
            )

        if args.freeze_num_iter > 0:
            for param in model.named_parameters():
                if param[0] == 'E.weight':
                    param[1].requires_grad = False

        if (self.opt.lower() == "adam") and (self.model != "poincare"):
            opt = Adam(
                model.parameters(),
                lr=args.freeze_entity_lr if args.freeze_entity_lr is not None else self.learning_rate,
            )
        else:
            opt = RiemannianSGD(
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=args.freeze_entity_lr if args.freeze_entity_lr is not None else self.learning_rate,
                param_names=[name for name, p in model.named_parameters() if p.requires_grad]
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        hits_at_10_per_iteration = []

        print("Starting training...")
        for it in range(1, self.num_iterations + 1):
            if it == args.freeze_num_iter + 1:
                for param in model.named_parameters():
                    if param[0] == 'E.weight':
                        param[1].requires_grad = True

                if (self.opt.lower() == "adam"):
                    opt = Adam(
                        list(filter(lambda p: p.requires_grad, model.parameters())),
                        lr=self.learning_rate
                    )
                else:
                    opt = RiemannianSGD(
                        list(filter(lambda p: p.requires_grad, model.parameters())),
                        lr=self.learning_rate,
                        param_names=[name for name, p in model.named_parameters() if p.requires_grad]
                    )

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

                if self.transe_loss:
                    pos_predictions = model.forward(
                        e1_idx[:, :1], r_idx[:, :1], e2_idx[:, :1]
                    )
                    neg_predictions = model.forward(
                        e1_idx[:, 1:], r_idx[:, 1:], e2_idx[:, 1:]
                    )
                    loss = model.loss(
                        pos_predictions.cpu(),
                        neg_predictions.cpu(),
                        targets[:, :1].cpu(),
                    )
                else:
                    predictions = model.forward(e1_idx, r_idx, e2_idx)
                    loss = model.loss(predictions.cpu(), targets.cpu())
                loss.backward()
                opt.step()
                losses.append(loss.item())
                if self.normalize_at_step:
                    model.E.weight.data = torch.nn.functional.normalize(
                        model.E.weight.data, p=2.0, dim=1, eps=1e-12, out=None
                    )
            print(it)
            print(time.time() - start_train)
            print(np.mean(losses))
            model.eval()
            with torch.no_grad():
                if not it % 5:
                    print("Test:")
                    last_hits_at_10 = self.evaluate(model, d.test_data)
                    hits_at_10_per_iteration.append(last_hits_at_10)
            
            if args.early_stopping:
                if len(hits_at_10_per_iteration) * 5 > args.early_stopping_patience:
                    if hits_at_10_per_iteration.index(max(hits_at_10_per_iteration)) < (it - args.early_stopping_patience) // 5:
                        print("Triggering early stopping")
                        print(f"Best iteration occurred around: {hits_at_10_per_iteration.index(max(hits_at_10_per_iteration)) * 5}")
                        break

        if args.save_model is not None:
            torch.save(model, os.path.join(args.save_model, f"{args.run_name}.pth"))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        "--normalize_rel_vec",
        type=str2bool,
        default=False,
        help="Normalize every relation vector to L2-norm=1",
    )
    parser.add_argument(
        "--relation_matrix_init",
        type=str,
        default=None,
        nargs="?",
        help="How to initialise relation matrix mure: baseline, glove, bert",
    )
    parser.add_argument(
        "--normalize_rel_mat",
        type=str2bool,
        default=False,
        help="Normalize every relation vector to L2-norm=1",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=40,
        nargs="?",
        help="Random seed for the pipeline",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=200,
        nargs="?",
        help="Embedding size",
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
        "--opt",
        type=str,
        default=None,
        nargs="?",
        help="Choose the optimizer for the training, only for euclidean model",
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
        type=str2bool,
        default=False,
        help="use entity descriptions to init BERT embs",
    )
    parser.add_argument(
        "--transe_arch",
        type=str2bool,
        default=False,
        help="Change MuRE architecture to make it as similar as possible to TransE",
    )
    parser.add_argument(
        "--transe_enable_bias",
        type=str2bool,
        default=False,
        help="Change MuRE architecture to make it as similar as possible to TransE",
    )
    parser.add_argument(
        "--transe_bias_mode",
        type=str,
        default="both",
        help="Specify the bias terms to include from mure",
    )
    parser.add_argument(
        "--transe_bias_init",
        type=str,
        default="zero",
        help="Specify the object bias term init distribution",
    )
    parser.add_argument(
        "--transe_enable_mtx",
        type=str2bool,
        default=False,
        help="Change MuRE architecture to make it as similar as possible to TransE",
    )
    parser.add_argument(
        "--transe_enable_vec",
        type=str2bool,
        default=False,
        help="Change MuRE architecture to only use multiplicative mechanisms",
    )
    parser.add_argument(
        "--mult_factor",
        type=float,
        default=None,
        nargs="?",
        help="Embedding size",
    )
    parser.add_argument(
        "--transe_loss",
        type=str2bool,
        default=False,
        help="Change MuRE loss to make it as similar as possible to TransE",
    )
    parser.add_argument(
        "--disable_inverse_triples",
        type=str2bool,
        default=False,
        help="Change MuRE architecture to make it as similar as possible to TransE",
    )
    parser.add_argument(
        "--normalize_entity_mtx",
        type=str2bool,
        default=False,
        help="Normalize every entity vector to L2-norm=1",
    )
    parser.add_argument(
        "--distmult_score_function",
        type=str2bool,
        default=False,
        help="Use different score function to compare with DistMult",
    )
    parser.add_argument(
        "--distmult_sqdist",
        type=str2bool,
        default=False,
        help="Intermediate score function between distmult and MuRE",
    )
    parser.add_argument(
        "--distmult_sqdist_mode",
        type=str,
        default="both",
        help="Specify the terms to keep from the mure norm",
    )
    parser.add_argument(
        "--normalize_at_step",
        type=str2bool,
        default=False,
        help="Intermediate score function between distmult and MuRE",
    )
    parser.add_argument(
        "--save_model",
        type=str,
        default=None,
        help="Indicate directory where to save the trained model",
    )
    parser.add_argument(
        "--bert_entity_and_relation",
        type=str2bool,
        default=False,
        help="Use both entity name and relation name as input to get entity embedding from bert",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        default=False,
        help="Use early stopping to stop training when validation loss stops increasing",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=30,
        help="Number of epochs to wait before early-stopping training",
    )
    parser.add_argument(
        "--freeze_num_iter",
        type=int,
        default=0,
        help="Number of iterations in which the entity matrix remains frozen at the start of training"
    )
    parser.add_argument(
        "--freeze_entity_lr",
        type=float,
        default=None,
        nargs="?",
        help="Training learning rate to keep while the entity matrix is frozen",
    )

    args = parser.parse_args()

    args.bert_layer = [int(x) for x in args.bert_layer]

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

    if args.model == "euclidian":
        model_run_name = "OgMuRE"
    elif args.model == "poincare":
        model_run_name = "OgMuRP"
    else:
        pass

    run_name = f"{args.init}_{args.dim}_{model_run_name}_{dataset}_opt_{args.opt}"

    if args.transe_arch:
        run_name = (
            f"{args.init}_{args.dim}_{model_run_name}_{dataset}_opt_{args.opt}_TransE"
        )
    if args.disable_inverse_triples:
        run_name += f"_NoInvT"
    if args.nneg < 50:
        run_name += f"_nneg_{args.nneg}"

    if args.distmult_score_function:
        args.transe_arch = True
        args.transe_enable_mtx = True
        args.transe_enable_vec = False

    if args.mult_factor is None:
        if args.transe_arch:
            args.mult_factor = 1
        else:
            args.mult_factor = 1e-3

    wandb.init(
        name=run_name,
        entity="eth_ai_center_kg_project",
        project="W2V_for_KGs",
        group=args.wandb_group,
    )
    wandb.config.use_pykeen = False
    wandb.config.update(args)

    args.run_name = run_name

    if args.distmult_sqdist and (args.distmult_sqdist_mode == "both"):
        args.distmult_sqdist_mode = ["subject", "object"]

    if args.transe_enable_bias and (args.transe_bias_mode == "both"):
        args.transe_bias_mode = ["subject", "object"]

    data_dir = "data/%s/" % dataset
    torch.backends.cudnn.deterministic = True
    seed = args.random_seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir, create_inverse_triples=not args.disable_inverse_triples)
    experiment = Experiment(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        dim=args.dim,
        cuda=args.cuda,
        nneg=args.nneg,
        model=args.model,
        opt=args.opt,
        transe_arch=args.transe_arch,
        transe_loss=args.transe_loss,
        mult_factor=args.mult_factor,
        transe_enable_bias=args.transe_enable_bias,
        transe_bias_mode=args.transe_bias_mode,
        transe_bias_init=args.transe_bias_init,
        transe_enable_mtx=args.transe_enable_mtx,
        transe_enable_vec=args.transe_enable_vec,
        distmult_score_function=args.distmult_score_function,
        distmult_sqdist=args.distmult_sqdist,
        distmult_sqdist_mode=args.distmult_sqdist_mode,
        normalize_at_step=args.normalize_at_step,
    )
    experiment.train_and_eval()
