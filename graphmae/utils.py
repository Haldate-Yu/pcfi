import os
import argparse
import torch
import torch.nn as nn
from functools import partial
import yaml
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError

    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias


def build_args():
    mae_parser = argparse.ArgumentParser(description="GMAE")
    mae_parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    mae_parser.add_argument("--mae_seeds", type=int, nargs="+", default=[42])
    mae_parser.add_argument("--dataset", type=str, default="cora")
    mae_parser.add_argument("--device", type=int, default=-1)
    mae_parser.add_argument("--max_epoch", type=int, default=200,
                            help="number of training epochs")
    mae_parser.add_argument("--warmup_steps", type=int, default=-1)

    mae_parser.add_argument("--num_heads", type=int, default=4,
                            help="number of hidden attention heads")
    mae_parser.add_argument("--num_out_heads", type=int, default=1,
                            help="number of output attention heads")
    mae_parser.add_argument("--num_layers", type=int, default=2,
                            help="number of hidden layers")
    mae_parser.add_argument("--num_hidden", type=int, default=256,
                            help="number of hidden units")
    mae_parser.add_argument("--residual", action="store_true", default=False,
                            help="use residual connection")
    mae_parser.add_argument("--in_drop", type=float, default=.2,
                            help="input feature dropout")
    mae_parser.add_argument("--attn_drop", type=float, default=.1,
                            help="attention dropout")
    mae_parser.add_argument("--norm", type=str, default=None)
    mae_parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
    mae_parser.add_argument("--weight_decay", type=float, default=5e-4,
                            help="weight decay")
    mae_parser.add_argument("--negative_slope", type=float, default=0.2,
                            help="the negative slope of leaky relu for GAT")
    mae_parser.add_argument("--activation", type=str, default="prelu")
    mae_parser.add_argument("--mask_rate", type=float, default=0.5)
    mae_parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    mae_parser.add_argument("--replace_rate", type=float, default=0.0)

    mae_parser.add_argument("--encoder", type=str, default="gat")
    mae_parser.add_argument("--decoder", type=str, default="gat")
    mae_parser.add_argument("--loss_fn", type=str, default="sce")
    mae_parser.add_argument("--alpha_l", type=float, default=2, help="`pow`coefficient for `sce` loss")
    mae_parser.add_argument("--optimizer", type=str, default="adam")

    mae_parser.add_argument("--max_epoch_f", type=int, default=30)
    mae_parser.add_argument("--lr_f", type=float, default=0.001, help="learning rate for evaluation")
    mae_parser.add_argument("--weight_decay_f", type=float, default=0.0, help="weight decay for evaluation")
    mae_parser.add_argument("--linear_prob", action="store_true", default=False)

    mae_parser.add_argument("--load_model", action="store_true")
    mae_parser.add_argument("--save_model", action="store_true")
    mae_parser.add_argument("--use_cfg", action="store_true")
    mae_parser.add_argument("--logging", action="store_true")
    mae_parser.add_argument("--scheduler", action="store_true", default=False)
    mae_parser.add_argument("--concat_hidden", action="store_true", default=False)

    # for graph classification
    mae_parser.add_argument("--pooling", type=str, default="mean")
    mae_parser.add_argument("--deg4feat", action="store_true", default=False, help="use node degree as input feature")
    mae_parser.add_argument("--batch_size", type=int, default=32)

    # for missing feature graphs
    mae_parser.add_argument(
        "--filling_method",
        type=str,
        help="Method to solve the missing feature problem",
        default="feature_propagation",
        choices=["random", "zero", "mean", "neighborhood_mean", "feature_propagation", "pcfi", "graphmae"],
    )
    mae_parser.add_argument(
        "--feature_init_type", type=str, help="Type of missing feature mask", default="zero",
        choices=["zero", "random"],
    )
    mae_parser.add_argument(
        "--feature_mask_type", type=str, help="Type of missing feature mask", default="uniform",
        choices=["uniform", "structural"],
    )
    # mae_parser.add_argument("--feature_missing_rate", type=float, help="Rate of node features missing", default=0.99)

    # load pretrained model
    mae_parser.add_argument("--pretrained_model_path", type=str)
    mae_parser.add_argument("--pretrained_model_name", type=str)
    mae_parser.add_argument("--task_type", type=str, default="transductive")
    mae_args = mae_parser.parse_known_args()[0]
    return mae_args


def load_best_configs(args, path):
    dataset = args.dataset.lower()

    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


def load_pretrained_model_path(mae_args, mae_seed):
    pretrained_model_dir = "./pretrain_gmae/" + mae_args.task_type + "/"
    file_name = mae_args.dataset.lower() + "_" + mae_args.encoder + "_" + mae_args.decoder + \
                "_" + mae_args.feature_init_type + "_" + mae_args.feature_mask_type + \
                "_" + str(mae_args.mask_rate) + "_" + str(mae_seed) + ".pt"
    mae_args.pretrained_model_name = file_name
    model_path = pretrained_model_dir + file_name
    if not os.path.exists(model_path):
        logging.info("Pretrained model not found")
        return None
    # logging.info("Loading pretrained model from " + model_path)
    # model = torch.load(model_path)
    return model_path
