"""
Copyright 2020 Twitter, Inc.
SPDX-License-Identifier: Apache-2.0
"""
import torch
import torch_sparse
from torch.nn.functional import normalize
from feature_propagation import FeaturePropagation
from pcfi import PCFI


def random_filling(X):
    return torch.randn_like(X)


def zero_filling(X):
    return torch.zeros_like(X)


def mean_filling(X, feature_mask):
    n_nodes = X.shape[0]
    return compute_mean(X, feature_mask).repeat(n_nodes, 1)


def neighborhood_mean_filling(edge_index, X, feature_mask):
    n_nodes = X.shape[0]
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    edge_values = torch.ones(edge_index.shape[1]).to(edge_index.device)
    edge_index_mm = torch.stack([edge_index[1], edge_index[0]]).to(edge_index.device)

    D = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, feature_mask.float())
    mean_neighborhood_features = torch_sparse.spmm(edge_index_mm, edge_values, n_nodes, n_nodes, X_zero_filled) / D
    # If a feature is not present on any neighbor, set it to 0
    mean_neighborhood_features[mean_neighborhood_features.isnan()] = 0

    return mean_neighborhood_features


def feature_propagation(edge_index, X, feature_mask, num_iterations):
    propagation_model = FeaturePropagation(num_iterations=num_iterations)

    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask)


def pcfi(edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None):
    propagation_model = PCFI(num_iterations=num_iterations, alpha=alpha, beta=beta)
    return propagation_model.propagate(x=X, edge_index=edge_index, mask=feature_mask, mask_type=mask_type)


def GraphMAE(model, edge_index, X, feature_mask, num_iterations=None, mask_type=None, use_only_encoder=False):
    if mask_type == "structural":
        mask_node_ids = torch.where(feature_mask.sum(dim=1) == 0)[0]
    elif mask_type == "uniform":
        mask_node_ids = torch.where(feature_mask.sum(dim=1) != feature_mask.shape[1])[0]

    node_mask = torch.ones(X.shape[0], dtype=torch.bool)
    node_mask[mask_node_ids] = False
    # return reconstructed_X as filled
    if use_only_encoder:
        pass
    else:
        return model.missing_attr_prediction(X, edge_index, node_mask, mask_type).detach()


def filling(filling_method, edge_index, X, feature_mask, num_iterations=None, mask_type=None, alpha=None, beta=None,
            pretrained_model=None, use_only_encoder=False, normalize_type=None):
    if filling_method == "random":
        X_reconstructed = random_filling(X)
    elif filling_method == "zero":
        X_reconstructed = zero_filling(X)
    elif filling_method == "mean":
        X_reconstructed = mean_filling(X, feature_mask)
    elif filling_method == "neighborhood_mean":
        X_reconstructed = neighborhood_mean_filling(edge_index, X, feature_mask)
    elif filling_method == "feature_propagation":
        X_reconstructed = feature_propagation(edge_index, X, feature_mask, num_iterations)
    elif filling_method == "pcfi":
        X_reconstructed = pcfi(edge_index, X, feature_mask, num_iterations, mask_type,
                               alpha, beta)
    elif filling_method == "graphmae":
        X_reconstructed = GraphMAE(pretrained_model, edge_index, X, feature_mask, num_iterations, mask_type,
                                   use_only_encoder)
        # normalize or gdc
        if normalize_type == "l1":
            X_reconstructed = normalize(X_reconstructed, p=1, dim=1)
        elif normalize_type == "l2":
            X_reconstructed = normalize(X_reconstructed, p=2, dim=1)

    else:
        raise ValueError(f"{filling_method} method not implemented")
    return X_reconstructed


def compute_mean(X, feature_mask):
    X_zero_filled = X
    X_zero_filled[~feature_mask] = 0.0
    num_of_non_zero = torch.count_nonzero(feature_mask, dim=0)
    mean_features = torch.sum(X_zero_filled, axis=0) / num_of_non_zero
    # If a feature is not present on any node, set it to 0
    mean_features[mean_features.isnan()] = 0

    return mean_features
