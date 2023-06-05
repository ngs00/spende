from torch.nn.functional import normalize
from torch_geometric.data import Data
from util.chem import *
from util.model import get_op_embs


def get_retrosyn_dataset(dataset, emb_net, elem_attrs):
    _op_embs = get_op_embs(dataset, emb_net)
    retrosyn_dataset = list()

    for n in range(0, len(dataset)):
        data = dataset.data[n]
        node_feats = list()
        targets = list()

        for mat in data.precursors:
            for e in mat.elem_dict:
                node_feats.append(elem_attrs[atom_nums[e] - 1, :])

                if e in data.products[0].elem_dict.keys():
                    targets.append(data.products[0].elem_dict[e])
                else:
                    targets.append(0)

        edges = [[i, j] for i in range(0, len(node_feats)) for j in range(0, len(node_feats))]
        node_feats = torch.vstack(node_feats)
        node_nums = torch.tensor(node_feats.shape[0], dtype=torch.long).view(1, 1)
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        op_embs = normalize(_op_embs[:, 0, :].flatten().unsqueeze(0), p=2, dim=1)
        targets = torch.tensor(targets, dtype=torch.float).view(-1, 1)
        retrosyn_dataset.append(Data(x=node_feats, edge_index=edges, op_embs=op_embs, node_nums=node_nums, y=targets))

    return retrosyn_dataset


def get_retrosyn_dataset_non_op(dataset, elem_attrs):
    retrosyn_dataset = list()

    for n in range(0, len(dataset)):
        data = dataset.data[n]
        node_feats = list()
        targets = list()

        for mat in data.precursors:
            print(mat.formula)
        print(data.products[0].formula)

        for mat in data.precursors:
            for e in mat.elem_dict:
                node_feats.append(elem_attrs[atom_nums[e] - 1, :])

                if e in data.products[0].elem_dict.keys():
                    targets.append(data.products[0].elem_dict[e])
                else:
                    targets.append(0)

        print(targets)
        print('------------------------------')

        edges = [[i, j] for i in range(0, len(node_feats)) for j in range(0, len(node_feats))]
        node_feats = torch.vstack(node_feats)
        node_nums = torch.tensor(node_feats.shape[0], dtype=torch.long).view(1, 1)
        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        targets = torch.tensor(targets, dtype=torch.float).view(-1, 1)
        retrosyn_dataset.append(Data(x=node_feats, edge_index=edges, node_nums=node_nums, y=targets))

    return retrosyn_dataset
