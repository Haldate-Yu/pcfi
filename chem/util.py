import argparse
import networkx as nx
import random
import torch
import torch.nn.functional as F
from rdkit.Chem import AllChem

from loader import MoleculeDataset
from loader import graph_data_obj_to_nx_simple, nx_to_graph_data_obj_simple


def load_args():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default='tox21',
                        help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default='', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument("--mae_seeds", type=int, nargs="+", default=[42])
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type=str, default="scaffold", help="random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default=0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--scheduler', action="store_true", default=False)

    # for missing feature graphs
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
    parser.add_argument(
        "--feature_mask_type", type=str, help="Type of missing feature mask", default="uniform",
        choices=["uniform", "structural"],
    )
    parser.add_argument("--feature_missing_rate", type=float, help="Rate of node features missing", default=0.99)

    parser.add_argument("--pretrained_model_path", type=str)

    return parser.parse_known_args()[0]


def check_same_molecules(s1, s2):
    mol1 = AllChem.MolFromSmiles(s1)
    mol2 = AllChem.MolFromSmiles(s2)
    return AllChem.MolToInchi(mol1) == AllChem.MolToInchi(mol2)


class NegativeEdge:
    def __init__(self):
        """
        Randomly sample negative edges
        """
        pass

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        edge_set = set([str(data.edge_index[0, i].cpu().item()) + "," + str(
            data.edge_index[1, i].cpu().item()) for i in
                        range(data.edge_index.shape[1])])

        redandunt_sample = torch.randint(0, num_nodes, (2, 5 * num_edges))
        sampled_ind = []
        sampled_edge_set = set([])
        for i in range(5 * num_edges):
            node1 = redandunt_sample[0, i].cpu().item()
            node2 = redandunt_sample[1, i].cpu().item()
            edge_str = str(node1) + "," + str(node2)
            if not edge_str in edge_set and not edge_str in sampled_edge_set and not node1 == node2:
                sampled_edge_set.add(edge_str)
                sampled_ind.append(i)
            if len(sampled_ind) == num_edges / 2:
                break

        data.negative_edge_index = redandunt_sample[:, sampled_ind]

        return data


class ExtractSubstructureContextPair:
    def __init__(self, k, l1, l2):
        """
        Randomly selects a node from the data object, and adds attributes
        that contain the substructure that corresponds to k hop neighbours
        rooted at the node, and the context substructures that corresponds to
        the subgraph that is between l1 and l2 hops away from the
        root node.
        :param k:
        :param l1:
        :param l2:
        """
        self.k = k
        self.l1 = l1
        self.l2 = l2

        # for the special case of 0, addresses the quirk with
        # single_source_shortest_path_length
        if self.k == 0:
            self.k = -1
        if self.l1 == 0:
            self.l1 = -1
        if self.l2 == 0:
            self.l2 = -1

    def __call__(self, data, root_idx=None):
        """

        :param data: pytorch geometric data object
        :param root_idx: If None, then randomly samples an atom idx.
        Otherwise sets atom idx of root (for debugging only)
        :return: None. Creates new attributes in original data object:
        data.center_substruct_idx
        data.x_substruct
        data.edge_attr_substruct
        data.edge_index_substruct
        data.x_context
        data.edge_attr_context
        data.edge_index_context
        data.overlap_context_substruct_idx
        """
        num_atoms = data.x.size()[0]
        if root_idx == None:
            root_idx = random.sample(range(num_atoms), 1)[0]

        G = graph_data_obj_to_nx_simple(data)  # same ordering as input data obj

        # Get k-hop subgraph rooted at specified atom idx
        substruct_node_idxes = nx.single_source_shortest_path_length(G,
                                                                     root_idx,
                                                                     self.k).keys()
        if len(substruct_node_idxes) > 0:
            substruct_G = G.subgraph(substruct_node_idxes)
            substruct_G, substruct_node_map = reset_idxes(substruct_G)  # need
            # to reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            substruct_data = nx_to_graph_data_obj_simple(substruct_G)
            data.x_substruct = substruct_data.x
            data.edge_attr_substruct = substruct_data.edge_attr
            data.edge_index_substruct = substruct_data.edge_index
            data.center_substruct_idx = torch.tensor([substruct_node_map[
                                                          root_idx]])  # need
            # to convert center idx from original graph node ordering to the
            # new substruct node ordering

        # Get subgraphs that is between l1 and l2 hops away from the root node
        l1_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l1).keys()
        l2_node_idxes = nx.single_source_shortest_path_length(G, root_idx,
                                                              self.l2).keys()
        context_node_idxes = set(l1_node_idxes).symmetric_difference(
            set(l2_node_idxes))
        if len(context_node_idxes) > 0:
            context_G = G.subgraph(context_node_idxes)
            context_G, context_node_map = reset_idxes(context_G)  # need to
            # reset node idx to 0 -> num_nodes - 1, otherwise data obj does not
            # make sense, since the node indices in data obj must start at 0
            context_data = nx_to_graph_data_obj_simple(context_G)
            data.x_context = context_data.x
            data.edge_attr_context = context_data.edge_attr
            data.edge_index_context = context_data.edge_index

        # Get indices of overlapping nodes between substruct and context,
        # WRT context ordering
        context_substruct_overlap_idxes = list(set(
            context_node_idxes).intersection(set(substruct_node_idxes)))
        if len(context_substruct_overlap_idxes) > 0:
            context_substruct_overlap_idxes_reorder = [context_node_map[old_idx]
                                                       for
                                                       old_idx in
                                                       context_substruct_overlap_idxes]
            # need to convert the overlap node idxes, which is from the
            # original graph node ordering to the new context node ordering
            data.overlap_context_substruct_idx = \
                torch.tensor(context_substruct_overlap_idxes_reorder)

        return data

        # ### For debugging ###
        # if len(substruct_node_idxes) > 0:
        #     substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
        #                                                  data.edge_index_substruct,
        #                                                  data.edge_attr_substruct)
        #     print(AllChem.MolToSmiles(substruct_mol))
        # if len(context_node_idxes) > 0:
        #     context_mol = graph_data_obj_to_mol_simple(data.x_context,
        #                                                data.edge_index_context,
        #                                                data.edge_attr_context)
        #     print(AllChem.MolToSmiles(context_mol))
        #
        # print(list(context_node_idxes))
        # print(list(substruct_node_idxes))
        # print(context_substruct_overlap_idxes)
        # ### End debugging ###

    def __repr__(self):
        return '{}(k={},l1={}, l2={})'.format(self.__class__.__name__, self.k,
                                              self.l1, self.l2)


def reset_idxes(G):
    """
    Resets node indices such that they are numbered from 0 to num_nodes - 1
    :param G:
    :return: copy of G with relabelled node indices, mapping
    """
    mapping = {}
    for new_idx, old_idx in enumerate(G.nodes()):
        mapping[old_idx] = new_idx
    new_G = nx.relabel_nodes(G, mapping, copy=True)
    return new_G, mapping


# TODO(Bowen): more unittests
class MaskAtom:
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=True):
        """
        Randomly masks an atom, and optionally masks edges connecting to it.
        The mask atom type index is num_possible_atom_type
        The mask edge type index in num_possible_edge_type
        :param num_atom_type:
        :param num_edge_type:
        :param mask_rate: % of atoms to be masked
        :param mask_edge: If True, also mask the edges that connect to the
        masked atoms
        """
        self.num_atom_type = num_atom_type
        self.num_edge_type = num_edge_type
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge

        self.num_chirality_tag = 3
        self.num_bond_direction = 3

    def __call__(self, data, masked_atom_indices=None):
        """

        :param data: pytorch geometric data object. Assume that the edge
        ordering is the default pytorch geometric ordering, where the two
        directions of a single edge occur in pairs.
        Eg. data.edge_index = tensor([[0, 1, 1, 2, 2, 3],
                                     [1, 0, 2, 1, 3, 2]])
        :param masked_atom_indices: If None, then randomly samples num_atoms
        * mask rate number of atom indices
        Otherwise a list of atom idx that sets the atoms to be masked (for
        debugging only)
        :return: None, Creates new attributes in original data object:
        data.mask_node_idx
        data.mask_node_label
        data.mask_edge_idx
        data.mask_edge_label
        """

        if masked_atom_indices == None:
            # sample x distinct atoms to be masked, based on mask rate. But
            # will sample at least 1 atom
            num_atoms = data.x.size()[0]
            sample_size = int(num_atoms * self.mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)

        # create mask node label by copying atom feature of mask atom
        mask_node_labels_list = []
        for atom_idx in masked_atom_indices:
            mask_node_labels_list.append(data.x[atom_idx].view(1, -1))
        data.mask_node_label = torch.cat(mask_node_labels_list, dim=0)
        data.masked_atom_indices = torch.tensor(masked_atom_indices)

        # ----------- graphMAE -----------
        atom_type = F.one_hot(data.mask_node_label[:, 0], num_classes=self.num_atom_type).float()
        atom_chirality = F.one_hot(data.mask_node_label[:, 1], num_classes=self.num_chirality_tag).float()
        # data.node_attr_label = torch.cat((atom_type,atom_chirality), dim=1)
        data.node_attr_label = atom_type

        # modify the original node feature of the masked node
        for atom_idx in masked_atom_indices:
            data.x[atom_idx] = torch.tensor([self.num_atom_type, 0])

        if self.mask_edge:
            # create mask edge labels by copying edge features of edges that are bonded to
            # mask atoms
            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(data.edge_index.cpu().numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in set((u, v)) and \
                            bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)

            if len(connected_edge_indices) > 0:
                # create mask edge labels by copying bond features of the bonds connected to
                # the mask atoms
                mask_edge_labels_list = []
                for bond_idx in connected_edge_indices[::2]:  # because the
                    # edge ordering is such that two directions of a single
                    # edge occur in pairs, so to get the unique undirected
                    # edge indices, we take every 2nd edge index from list
                    mask_edge_labels_list.append(
                        data.edge_attr[bond_idx].view(1, -1))

                data.mask_edge_label = torch.cat(mask_edge_labels_list, dim=0)
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    data.edge_attr[bond_idx] = torch.tensor(
                        [self.num_edge_type, 0])

                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices[::2])
            else:
                data.mask_edge_label = torch.empty((0, 2)).to(torch.int64)
                data.connected_edge_indices = torch.tensor(
                    connected_edge_indices).to(torch.int64)

            edge_type = F.one_hot(data.mask_edge_label[:, 0], num_classes=self.num_edge_type).float()
            bond_direction = F.one_hot(data.mask_edge_label[:, 1], num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_direction), dim=1)
            # data.edge_attr_label = edge_type

        return data

    def __repr__(self):
        return '{}(num_atom_type={}, num_edge_type={}, mask_rate={}, mask_edge={})'.format(
            self.__class__.__name__, self.num_atom_type, self.num_edge_type,
            self.mask_rate, self.mask_edge)


def get_missing_feature_mask(rate, n_nodes, n_features, type="uniform"):
    """
    Return mask of shape [n_nodes, n_features] indicating whether each feature is present or missing.
    If `type`='uniform', then each feature of each node is missing uniformly at random with probability `rate`.
    Instead, if `type`='structural', either we observe all features for a node, or we observe none. For each node
    there is a probability of `rate` of not observing any feature.
    """
    if type == "structural":  # either remove all of a nodes features or none
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes)).bool().unsqueeze(1).repeat(1, n_features)
    elif type == "uniform":
        return torch.bernoulli(torch.Tensor([1 - rate]).repeat(n_nodes, n_features)).bool()


if __name__ == "__main__":
    transform = NegativeEdge()
    dataset = MoleculeDataset("dataset/tox21", dataset="tox21")
    transform(dataset[0])

    """
    # TODO(Bowen): more unit tests
    # test ExtractSubstructureContextPair

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    data = mol_to_graph_data_obj_simple(m)
    root_idx = 13

    # 0 hops: no substructure or context. We just test the absence of x attr
    transform = ExtractSubstructureContextPair(0, 0, 0)
    transform(data, root_idx)
    assert not hasattr(data, 'x_substruct')
    assert not hasattr(data, 'x_context')

    # k > n_nodes, l1 = 0 and l2 > n_nodes: substructure and context same as
    # molecule
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(100000, 0, 100000)
    transform(data, root_idx)
    substruct_mol = graph_data_obj_to_mol_simple(data.x_substruct,
                                                 data.edge_index_substruct,
                                                 data.edge_attr_substruct)
    context_mol = graph_data_obj_to_mol_simple(data.x_context,
                                               data.edge_index_context,
                                               data.edge_attr_context)
    assert check_same_molecules(AllChem.MolToSmiles(substruct_mol),
                                AllChem.MolToSmiles(context_mol))

    transform = ExtractSubstructureContextPair(1, 1, 10000)
    transform(data, root_idx)

    # increase k from 0, and increase l1 from 1 while keeping l2 > n_nodes: the
    # total number of atoms should be n_atoms
    for i in range(len(m.GetAtoms())):
        data = mol_to_graph_data_obj_simple(m)
        print('i: {}'.format(i))
        transform = ExtractSubstructureContextPair(i, i, 100000)
        transform(data, root_idx)
        if hasattr(data, 'x_substruct'):
            n_substruct_atoms = data.x_substruct.size()[0]
        else:
            n_substruct_atoms = 0
        print('n_substruct_atoms: {}'.format(n_substruct_atoms))
        if hasattr(data, 'x_context'):
            n_context_atoms = data.x_context.size()[0]
        else:
            n_context_atoms = 0
        print('n_context_atoms: {}'.format(n_context_atoms))
        assert n_substruct_atoms + n_context_atoms == len(m.GetAtoms())

    # l1 < k and l2 >= k, so an overlap exists between context and substruct
    data = mol_to_graph_data_obj_simple(m)
    transform = ExtractSubstructureContextPair(2, 1, 3)
    transform(data, root_idx)
    assert hasattr(data, 'center_substruct_idx')

    # check correct overlap atoms between context and substruct


    # m = AllChem.MolFromSmiles('COC1=CC2=C(NC(=N2)[S@@](=O)CC2=NC=C(C)C(OC)=C2C)C=C1')
    # data = mol_to_graph_data_obj_simple(m)
    # root_idx = 9
    # k = 1
    # l1 = 1
    # l2 = 2
    # transform = ExtractSubstructureContextPaidata = mol_to_graph_data_obj_simple(m)r(k, l1, l2)
    # transform(data, root_idx)
    pass

    # TODO(Bowen): more unit tests
    # test MaskAtom
    from loader import mol_to_graph_data_obj_simple, \
        graph_data_obj_to_mol_simple

    smiles = 'C#Cc1c(O)c(Cl)cc(/C=C/N)c1S'
    m = AllChem.MolFromSmiles(smiles)
    original_data = mol_to_graph_data_obj_simple(m)
    num_atom_type = 118
    num_edge_type = 5

    # manually specify masked atom indices, don't mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=False)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    assert not hasattr(data, 'mask_edge_label')
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()

    # manually specify masked atom indices, mask edge
    masked_atom_indices = [13, 12]
    data = mol_to_graph_data_obj_simple(m)
    transform = MaskAtom(num_atom_type, num_edge_type, 0.1, mask_edge=True)
    transform(data, masked_atom_indices)
    assert data.mask_node_label.size() == torch.Size(
        (len(masked_atom_indices), 2))
    # check that the correct rows in x have been modified to be mask atom type
    assert (data.x[masked_atom_indices] == torch.tensor(([num_atom_type,
                                                          0]))).all()
    assert (data.mask_node_label == original_data.x[masked_atom_indices]).all()
    # check that the correct rows in edge_attr have been modified to be mask edge
    # type, and the mask_edge_label are correct
    rdkit_bonds = []
    for atom_idx in masked_atom_indices:
        bond_indices = list(AllChem.FindAtomEnvironmentOfRadiusN(m, radius=1,
                                                                 rootedAtAtom=atom_idx))
        for bond_idx in bond_indices:
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetBeginAtomIdx(), m.GetBonds()[
                    bond_idx].GetEndAtomIdx()))
            rdkit_bonds.append(
                (m.GetBonds()[bond_idx].GetEndAtomIdx(), m.GetBonds()[
                    bond_idx].GetBeginAtomIdx()))
    rdkit_bonds = set(rdkit_bonds)
    connected_edge_indices = []
    for i in range(data.edge_index.size()[1]):
        if tuple(data.edge_index.numpy().T[i].tolist()) in rdkit_bonds:
            connected_edge_indices.append(i)
    assert (data.edge_attr[connected_edge_indices] ==
            torch.tensor(([num_edge_type, 0]))).all()
    assert (data.mask_edge_label == original_data.edge_attr[
        connected_edge_indices[::2]]).all() # data.mask_edge_label contains
    # the unique edges (ignoring direction). The data obj has edge ordering
    # such that two directions of a single edge occur in pairs, so to get the
    # unique undirected edge indices, we take every 2nd edge index from list
    """
