import itertools
import torch_geometric.data as tgd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from chemparse import parse_formula
from util.chem import *
from util.operation import *


class Material:
    def __init__(self, formula, fvec, amount):
        self.formula = formula
        self.fvec = fvec.view(1, -1)
        self.amount = torch.tensor(amount, dtype=torch.float).view(1, 1)
        self.elem_dict = parse_formula(self.formula)
        self.formula_host = ''

        for e in self.elem_dict.keys():
            if self.elem_dict[e] >= 1.0:
                self.formula_host += e + str(self.elem_dict[e])


class Data:
    def __init__(self, precursors, products, operations, mat_properties, doi, idx):
        self.precursors = precursors
        self.products = products
        self.operations = operations
        self.elem_graph = self.__get_elem_graph()
        self.mat_properties = mat_properties
        self.doi = doi
        self.idx = idx

    def __get_elem_graph(self):
        nodes = self.precursors + self.products
        node_feats = torch.vstack([mat.fvec for mat in nodes])
        edges = list()

        for i in range(0, len(self.precursors)):
            edges.append([i, i])

            for j in range(len(self.precursors), len(nodes)):
                edges.append([i, j])
                edges.append([j, i])
                edges.append([j, j])

        edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
        fvec = torch.mean(node_feats, dim=0).view(1, -1)

        return tgd.Data(x=node_feats, edge_index=edges, fvec=fvec)


class Dataset:
    def __init__(self, data, max_operations):
        self.data = data
        self.elem_graphs = [mat.elem_graph for mat in self.data]
        self.labels = list()
        self.n_classes = len(op_types.keys())
        self.max_operations = max_operations
        self.dim_elem_feats = self.elem_graphs[0].x.shape[1]
        self.forms = [d.products[0].formula for d in data]
        self.idx_data = [d.idx for d in data]

        for d in self.data:
            dlabels = list()

            for op in d.operations:
                dlabels.append(op.label)

            for i in range(0, self.max_operations - len(d.operations)):
                dlabels.append(op_types['EndOperation'])
            self.labels.append(dlabels)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.elem_graphs[idx], self.labels[idx]


def load_dataset(path_dataset, elem_attrs):
    list_data = list()

    with open(path_dataset, encoding='utf-8') as f:
        data = json.load(f)

    for i in tqdm(range(0, len(data))):
        list_precursors = list()
        list_products = list()
        list_operations = list()

        try:
            precursors = data[i]['precursors']
            products = data[i]['target']['composition']
            operations = data[i]['operations']

            # Read precursors.
            for prec in precursors:
                for mat in prec['composition']:
                    formula = mat['formula']
                    elem_dict = mat['elements']
                    amount = float(mat['amount'])
                    fvec = get_fvec(elem_dict, elem_attrs)
                    list_precursors.append(Material(formula, fvec, amount))

            # Read products.
            for mat in products:
                formula = mat['formula']
                elem_dict = mat['elements']
                amount = float(mat['amount'])
                fvec = get_fvec(elem_dict, elem_attrs)
                list_products.append(Material(formula, fvec, amount))

            # Read operations.
            if len(operations) == 0:
                raise ValueError('Empty operations in the {}-th data'.format(i))

            list_operations.append(Operation('start', 'StartOperation', None, op_types['StartOperation']))
            for op in operations:
                op_name = op['string']
                op_type = op['type']
                conditions = op['conditions']
                label = op_types[op_type]
                list_operations.append(Operation(op_name, op_type, conditions, label))
            list_operations.append(Operation('end', 'EndOperation', None, op_types['EndOperation']))
            list_data.append(Data(list_precursors, list_products, list_operations,
                                  data[i]['material property'], data[i]['doi'], i))

        except ValueError as e:
            print(e)

    return Dataset(list_data, numpy.max([len(d.operations) for d in list_data]))


def split_list(list_obj, ratio_train, random_seed=None):
    n_data1 = int(ratio_train * len(list_obj))

    if random_seed is not None:
        numpy.random.seed(random_seed)

    idx_rand = numpy.random.permutation(len(list_obj))

    list1 = [list_obj[idx] for idx in idx_rand[:n_data1]]
    list2 = [list_obj[idx] for idx in idx_rand[n_data1:]]

    return list1, list2


def get_k_folds(dataset, k, random_seed=None):
    if random_seed is not None:
        numpy.random.seed(random_seed)

    k_folds = list()
    idx_rand = numpy.array_split(numpy.random.permutation(len(dataset)), k)

    for i in range(0, k):
        idx_train = list(itertools.chain.from_iterable(idx_rand[:i] + idx_rand[i+1:]))
        idx_test = idx_rand[i]
        dataset_train = Dataset([dataset.data[idx] for idx in idx_train], dataset.max_operations)
        dataset_test = Dataset([dataset.data[idx] for idx in idx_test], dataset.max_operations)
        k_folds.append([dataset_train, dataset_test])

    return k_folds


def collate(batch):
    return Batch.from_data_list([b[0] for b in batch]), torch.vstack([b[1] for b in batch])


def get_pred_dataset(dataset, target_name, model):
    x = numpy.vstack([d.products[0].fvec for d in dataset.data])
    temperature = numpy.vstack([d.mat_properties['temperature (K)'] for d in dataset.data])
    y = numpy.vstack([d.mat_properties[target_name] for d in dataset.data])

    if model is not None:
        data_loader = DataLoader(dataset, batch_size=128, collate_fn=collate)
        embs = model.emb(data_loader).numpy()
        x = numpy.hstack([x, embs[2], embs[3], embs[4]])

    return numpy.hstack([x, temperature]), y
