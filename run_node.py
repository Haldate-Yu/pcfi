"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import numpy as np
import argparse
import torch
from torch_geometric.data import NeighborSampler
import logging
import time
from data_loading import get_dataset
from data_utils import set_train_val_test_split
from graphmae import build_model
from chem.model import GNN_graphpred
from utils import get_missing_feature_mask, save_node_results
from models import get_model
from seeds import seeds
from evaluation import test
from train import train
import random
from filling_strategies import filling
from pcfi import pcfi
import warnings

# ignore user warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser("Setting for graphs with partially known features")
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of dataset",
    default="Cora",
    choices=[
        "Cora",
        "CiteSeer",
        "PubMed",
        "OGBN-Arxiv",
        "OGBN-Products",
        "MixHopSynthetic",
        "Photo",
        "Computers",
    ],
)
parser.add_argument(
    "--mask_type", type=str, help="Type of missing feature mask", default="structural",
    choices=["uniform", "structural"],
)
parser.add_argument(
    "--filling_method",
    type=str,
    help="Method to solve the missing feature problem",
    default="feature_propagation",
    choices=["random", "zero", "mean", "neighborhood_mean", "feature_propagation", "pcfi", "graphmae"],
)
parser.add_argument(
    "--feature_init_type", type=str, help="Type of missing feature mask", default="zero",
    choices=["zero", "random"],
)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.9)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument(
    "--model",
    type=str,
    help="Type of model to make a prediction on the downstream task",
    default="gcn",
    choices=["mlp", "sgc", "sage", "gcn", "gat", "gcnmf", "pagnn", "lp"],
)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.005)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=3)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")
parser.add_argument(
    "--batch_size", type=int, help="Batch size for models trained with neighborhood sampling", default=1024,
)
parser.add_argument(
    "--graph_sampling",
    help="Set if you want to use graph sampling (always true for large graphs)",
    action="store_true",
)
parser.add_argument(
    "--homophily", type=float, help="Level of homophily for synthetic datasets", default=None,
)
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument(
    "--log", type=str, help="Log Level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
)
# MAE args
parser.add_argument("--mae_missing_rate", type=float, default=0.5)
parser.add_argument("--task_type", type=str, default="transductive")
parser.add_argument("--mae_seeds", type=int, nargs="+", default=[42])
parser.add_argument("--pretrained_model_path", type=str)
parser.add_argument("--use_only_encoder", action="store_true", help="Whether to use only encoder of GMAE")


def run(args, graphmae_args=None, mae_seed=None):
    logger.info(args)

    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() else "cpu"
    )

    if args.filling_method == "graphmae":
        dataset, evaluator = get_dataset(name=args.dataset_name, use_lcc=False, homophily=args.homophily)
    else:
        dataset, evaluator = get_dataset(name=args.dataset_name, homophily=args.homophily)

    split_idx = dataset.get_idx_split() if hasattr(dataset, "get_idx_split") else None
    n_nodes, n_features = dataset.data.x.shape
    test_accs, best_val_accs, train_times = [], [], []

    for seed in seeds[: args.n_runs]:
        # setting seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

        train_loader = (
            NeighborSampler(
                dataset.data.edge_index,
                node_idx=split_idx["train"],
                sizes=[15, 10, 5][: args.num_layers],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=12,
            )
            if args.graph_sampling
            else None
        )
        # Setting `sizes` to -1 simply loads all the neighbors for each node. We can do this while evaluating
        # as we first compute the representation of all nodes after the first layer (in batches), then for the second layer, and so on
        inference_loader = (
            NeighborSampler(
                dataset.data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12,
            )
            if args.graph_sampling
            else None
        )

        num_classes = dataset.num_classes
        data = (set_train_val_test_split(
            seed=seed, data=dataset.data, split_idx=split_idx, dataset_name=args.dataset_name, )
                .to(device))
        train_start = time.time()
        if args.model == "lp":
            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=None,
                args=args,
            ).to(device)
            logger.info("Starting Label Propagation")
            logits = model(y=data.y, edge_index=data.edge_index, mask=data.train_mask)
            (_, val_acc, test_acc), _ = test(model=None, x=None, data=data, logits=logits, evaluator=evaluator)
        else:
            missing_feature_mask = get_missing_feature_mask(
                rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed=seed, type=args.mask_type,
            ).to(device)
            x = data.x.clone()
            pretrained_gmae = None
            if args.filling_method == "graphmae":
                graphmae_args.num_features = n_features
                graphMAE = build_model(graphmae_args)
                # loading pre-trained model
                # model_params = graphMAE.state_dict()
                # print("original model params")
                # for model_key, model_value in model_params.items():
                #     print(model_key)
                #
                # print("loaded model params")
                # state_dict = torch.load(graphmae_args.pretrained_model_path)
                # for key in list(state_dict.keys()):
                #     print(key)

                graphMAE.load_state_dict(torch.load(graphmae_args.pretrained_model_path, weights_only=True),
                                         strict=False)
                pretrained_gmae = graphMAE.to(device)
                # GMAE feature init method
                # todo define as a refinement

                if graphmae_args.feature_init_type == "zero":
                    x[~missing_feature_mask] = float(0)
                elif graphmae_args.feature_init_type == "random":
                    init_x = torch.randn_like(x)
                    x[~missing_feature_mask] = init_x[~missing_feature_mask]
                else:
                    raise ValueError(f"{args.feature_init_type} not implemented!")
            else:
                x[~missing_feature_mask] = float("nan")

            logger.debug("Starting feature filling")
            start = time.time()
            # todo fill with GMAE
            filled_features = (
                filling(args.filling_method, data.edge_index, x, missing_feature_mask, args.num_iterations,
                        args.mask_type, args.alpha, args.beta, pretrained_gmae, args.use_only_encoder)
                if args.model not in ["gcnmf", "pagnn"]
                else torch.full_like(x, float("nan"))
            )
            logger.debug(f"Feature filling completed. It took: {time.time() - start:.2f}s")

            model = get_model(
                model_name=args.model,
                num_features=data.num_features,
                num_classes=num_classes,
                edge_index=data.edge_index,
                x=x,
                mask=missing_feature_mask,
                args=args,
            ).to(device)
            params = list(model.parameters())

            optimizer = torch.optim.Adam(params, lr=args.lr)
            criterion = torch.nn.NLLLoss()
            test_acc = 0
            val_accs = []
            for epoch in range(0, args.epochs):
                x = torch.where(missing_feature_mask, data.x, filled_features)
                train(
                    model, x, data, optimizer, criterion, train_loader=train_loader, device=device,
                )
                (train_acc, val_acc, tmp_test_acc), out = test(
                    model, x=x, data=data, evaluator=evaluator, inference_loader=inference_loader, device=device,
                )
                if epoch == 0 or val_acc > max(val_accs):
                    test_acc = tmp_test_acc
                    y_soft = out.softmax(dim=-1)
                val_accs.append(val_acc)
                if epoch > args.patience and max(val_accs[-args.patience:]) <= max(val_accs[: -args.patience]):
                    break
                logger.debug(
                    f"Epoch {epoch + 1} - Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f}, Test acc: {tmp_test_acc:.3f}. It took {time.time() - start:.2f}s"
                )
            # todo following fp?
            # (_, val_acc, test_acc), _ = test(model, x=x, data=data, logits=y_soft, evaluator=evaluator)
        best_val_accs.append(val_acc)
        test_accs.append(test_acc)
        train_times.append(time.time() - train_start)

    test_acc_mean, test_acc_std = np.mean(test_accs), np.std(test_accs)
    logging.info(f"Test Accuracy: {test_acc_mean * 100:.2f} ± {test_acc_std * 100:.2f}\n")
    # save to file
    save_node_results(args, graphmae_args, test_acc_mean, test_acc_std)


if __name__ == "__main__":
    args = parser.parse_known_args()[0]
    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))
    logging.info(f"Running with args: {args}\n")
    # load graphMAE args
    if args.filling_method == "graphmae":
        from graphmae.utils import build_args, load_best_configs, load_pretrained_model_path

        graphmae_args = build_args()
        # loading mae args according to args
        graphmae_args.mae_seeds = args.mae_seeds
        graphmae_args.feature_init_type = args.feature_init_type
        graphmae_args.task_type = args.task_type
        graphmae_args.dataset = args.dataset_name
        # graphmae_args.feature_mask_type = args.mask_type
        graphmae_args.filling_method = args.filling_method
        graphmae_args.missing_rate = args.mae_missing_rate
        if graphmae_args.use_cfg:
            graphmae_args = load_best_configs(graphmae_args, "configs.yml")
        elif graphmae_args.use_high_missing_cfg:
            graphmae_args = load_best_configs(graphmae_args, "high_missing_configs.yml")


        for mae_seed in graphmae_args.mae_seeds:
            logging.info(f"Running graphMAE with seed: {mae_seed}\n")
            if args.pretrained_model_path is None:
                graphmae_args.pretrained_model_path = load_pretrained_model_path(graphmae_args, mae_seed)
            else:
                graphmae_args.pretrained_model_path = args.pretrained_model_path
                graphmae_args.pretrained_model_name = args.pretrained_model_path
            logging.info(f"Using graphMAE with args: {graphmae_args}\n")
            logging.info(f"Using graphMAE with pre-trained model: {graphmae_args.pretrained_model_path}\n")

            if graphmae_args.pretrained_model_path is None:
                logging.info("No pre-trained model found. Please specify the path using --pretrained_model_path")
                continue

            run(args, graphmae_args, mae_seed)
    else:
        run(args, None)
