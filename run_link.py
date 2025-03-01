"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
Modified by Daeho Um (daehoum1@snu.ac.kr)
"""
import time
import numpy as np
from tqdm import tqdm
import argparse
import torch
from data_loading import get_dataset
from graphmae import build_model
from utils import get_missing_feature_mask, save_link_results
from seeds import seeds
from utils_link import train, test
import random
from torch_geometric.utils import train_test_split_edges
from models import GCNEncoder
from torch_geometric.nn import GAE
import logging
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
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--missing_rate", type=float, help="Rate of node features missing", default=0.9)
parser.add_argument(
    "--num_iterations", type=int, help="Number of diffusion iterations for feature reconstruction", default=100,
)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=200)
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("--epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--n_runs", type=int, help="Max number of runs", default=10)
parser.add_argument("--hidden_dim", type=int, help="Hidden dimension of model", default=64)
parser.add_argument("--jk", action="store_true", help="Whether to use the jumping knowledge scheme")
parser.add_argument(
    "--log", type=str, help="Log Level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
)
# MAE args
parser.add_argument("--mae_missing_rate", type=float, default=0.5)
parser.add_argument("--use_cfg", action="store_true")
parser.add_argument("--task_type", type=str, default="transductive")
parser.add_argument("--mae_seeds", type=int, nargs="+", default=[42])
parser.add_argument("--pretrained_model_path", type=str)


def run(args, graphmae_args=None, mae_seed=None):
    device = torch.device(
        f"cuda:{args.gpu_idx}"
        if torch.cuda.is_available() and not (args.dataset_name == "OGBN-Products" and args.model == "lp")
        else "cpu"
    )

    dataset, evaluator = get_dataset(name=args.dataset_name)
    n_nodes, n_features = dataset.data.x.shape
    aucs, aps = [], []

    for seed in tqdm(seeds[: args.n_runs]):
        # setting seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dataset, evaluator = get_dataset(name=args.dataset_name)
        data = dataset.data

        data = train_test_split_edges(data, val_ratio=0.05, test_ratio=0.1).to(device)
        missing_feature_mask = get_missing_feature_mask(
            rate=args.missing_rate, n_nodes=n_nodes, n_features=n_features, seed=seed, type=args.mask_type,
        ).to(device)
        x = data.x.clone()
        pretrained_gmae = None
        if args.filling_method == "graphmae":
            graphmae_args.num_features = n_features
            graphMAE = build_model(graphmae_args)
            # todo loading pre-trained model
            model_params = graphMAE.state_dict()
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

        # filled_features = pcfi(data.train_pos_edge_index, x, missing_feature_mask, args.num_iterations, args.mask_type,
        #                        args.alpha, args.beta).to(device)
        filled_features = (
            filling(args.filling_method, data.train_pos_edge_index, x, missing_feature_mask, args.num_iterations,
                    args.mask_type, args.alpha, args.beta, pretrained_gmae).to(device)
        )

        data = data.to(device)
        x = torch.where(missing_feature_mask, data.x, filled_features)
        data.train_mask = data.val_mask = data.test_mask = None

        model = GAE(GCNEncoder(dataset.num_features, out_channels=16))
        model.to(device)
        train_pos_edge_index = data.train_pos_edge_index
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        epochs = 200
        for epoch in range(1, epochs + 1):
            loss = train(model, x, train_pos_edge_index, optimizer)
            auc, ap = test(model, x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
            if epoch % 50 == 0:
                print('epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

        aucs.append(auc)
        aps.append(ap)
    test_auc_mean, test_auc_std = np.mean(aucs), np.std(aucs)
    test_ap_mean, test_ap_std = np.mean(aps), np.std(aps)

    print(f"AUC Accuracy: {test_auc_mean * 100:.2f} ± {test_auc_std * 100:.2f}")
    print(f"AP Accuracy: {test_ap_mean * 100:.2f} ± {test_ap_std * 100:.2f}\n")
    # save to files
    save_link_results(args, graphmae_args, test_auc_mean, test_auc_std, test_ap_mean, test_ap_std)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    logger = logging.getLogger(__name__)
    logger.setLevel(level=getattr(logging, args.log.upper(), None))
    # load graphMAE args
    if args.filling_method == "graphmae":
        from graphmae.utils import build_args, load_best_configs, load_pretrained_model_path

        graphmae_args = build_args()
        # loading mae args according to args
        graphmae_args.use_cfg = args.use_cfg
        graphmae_args.mae_seeds = args.mae_seeds
        graphmae_args.feature_init_type = args.feature_init_type
        graphmae_args.task_type = args.task_type
        graphmae_args.dataset = args.dataset_name
        graphmae_args.feature_mask_type = args.mask_type
        graphmae_args.filling_method = args.filling_method
        graphmae_args.missing_rate = args.mae_missing_rate
        if graphmae_args.use_cfg:
            graphmae_args = load_best_configs(graphmae_args, "configs.yml")


        for mae_seed in graphmae_args.mae_seeds:
            if args.pretrained_model_path is None:
                graphmae_args.pretrained_model_path = load_pretrained_model_path(graphmae_args, mae_seed)
            else:
                graphmae_args.pretrained_model_path = args.pretrained_model_path
                graphmae_args.pretrained_model_name = args.pretrained_model_path
            logging.info(f"Using graphMAE with args: {graphmae_args}")
            logging.info(f"Using graphMAE with pre-trained model: {graphmae_args.pretrained_model_path}\n")

            if graphmae_args.pretrained_model_path is None:
                logging.info("No pre-trained model found. Please specify the path using --pretrained_model_path")
                continue

            run(args, graphmae_args, mae_seed)
    else:
        run(args, None)
