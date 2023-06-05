from util.model import *


valid_ops = [
    'HeatingOperation',
    'CoolingOperation',
    'GrindingOperation',
    'SinteringOperation',
    'PressingOperation'
]


class CondDataset:
    def __init__(self, data):
        self.prod_graph = list()
        self.query = list()
        self.op_emb = list()
        self.y = list()

        for d in data:
            self.prod_graph.append(d[0])
            self.query.append(d[1])
            self.op_emb.append(d[2])
            self.y.append(d[3])
        self.query = torch.vstack(self.query)
        self.op_emb = torch.vstack(self.op_emb)
        self.y = torch.tensor(self.y, dtype=torch.float).view(-1, 1)

    def __len__(self):
        return len(self.prod_graph)

    def __getitem__(self, idx):
        return self.prod_graph[idx], self.query[idx], self.op_emb[idx], self.y[idx]


def get_cond_dataset(dataset_syn, model):
    data_loader = DataLoader(dataset_syn, batch_size=128, collate_fn=collate)
    embs = model.emb(data_loader)
    list_data = {op + '_' + cond: list() for op in valid_ops for cond in op_conds[op]}
    datasets = dict()

    for n in range(0, len(dataset_syn)):
        data = dataset_syn.data[n]

        for t in range(0, len(data.operations)):
            op = data.operations[t]

            if op.label_cond is None:
                continue

            for cond in op_conds[op.op_type].keys():
                if op.label_cond[cond] is not None:
                    feat_vec = numpy.hstack([data.products[0].fvec.squeeze(0), embs[t, n]])
                    target = op.label_cond[cond]
                    list_data[op.op_type + '_' + cond].append(numpy.hstack([feat_vec, target]))

    for task in list_data.keys():
        task_info = task.split('_')
        cond_vals = op_conds[task_info[0]][task_info[1]]

        if isinstance(cond_vals, list):
            if len(cond_vals) == 2:
                datasets[task] = (numpy.vstack(list_data[task]), 'bin_clf')
            else:
                datasets[task] = (numpy.vstack(list_data[task]), 'multi_clf')
        else:
            datasets[task] = (numpy.vstack(list_data[task]), 'reg')

    return datasets


def load_emb_net(path_model_file, elem_attrs, dim_hidden_dec, dim_enc, dim_hidden):
    dataset = load_dataset('dataset/tmsr.json', elem_attrs)
    decoder = Decoder(dataset.n_classes, dim_hidden_dec, n_layers=2, dropout=0.6)
    model = Seq2Seq(dataset, dim_enc, decoder, dim_hidden, len_seq=dataset.max_operations)
    model.load_state_dict(torch.load(path_model_file))

    return model.cuda()
