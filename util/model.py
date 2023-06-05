import torch
import random
import numpy
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch_geometric.nn.conv import GATv2Conv, GINConv
from torch_geometric.nn.glob import global_mean_pool
from util.data import load_dataset, collate
from util.operation import op_types, op_conds


class Encoder(nn.Module):
    def __init__(self, dim_elem_feats, dim_enc):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.gnn = GATv2Conv(dim_elem_feats, 64, heads=4, concat=False)
        # self.gnn = GINConv(nn.Linear(dim_elem_feats, 64))
        self.fc = nn.Linear(dim_enc + 64, dim_enc)

    def forward(self, hy, g):
        x = self.dropout(g.x)
        hg = global_mean_pool(f.leaky_relu(self.gnn(x, g.edge_index)), g.batch)
        hg = hg.unsqueeze(0)
        h = self.fc(torch.cat([hy, hg], dim=2))

        return h


class Decoder(nn.Module):
    def __init__(self, n_classes, dim_hidden_dec, n_layers, dropout):
        super(Decoder, self).__init__()
        self.n_classes = n_classes
        self.dim_hidden_dec = dim_hidden_dec
        self.n_layers = n_layers
        self.embedding = nn.Embedding(n_classes, dim_hidden_dec)
        self.rnn = nn.GRU(dim_hidden_dec, dim_hidden_dec, n_layers, dropout=dropout)
        self.fc = nn.Linear(dim_hidden_dec, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, emb, h):
        z, hidden = self.rnn(emb, h)
        out = self.fc(z)

        return out, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dim_hidden, len_seq):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embedding = nn.Embedding(self.decoder.n_classes, dim_hidden)
        self.dropout = nn.Dropout(p=0.6)
        self.n_classes = decoder.fc.weight.shape[0]
        self.len_seq = len_seq

    def forward(self, g, y):
        yl = self.dropout(self.embedding(torch.swapaxes(y[0, :].view(-1, 1), 0, 1)))
        h = self.encoder(yl, g).repeat(2, 1, 1)
        out = torch.empty((self.len_seq, y.shape[1], self.n_classes))

        for t in range(0, self.len_seq):
            _y, h = self.decoder(yl, h)
            yl = torch.argmax(_y.squeeze(0), dim=1) if random.random() < 0.5 else y[t, :]
            yl = self.dropout(self.embedding(torch.swapaxes(yl.view(-1, 1), 0, 1)))
            out[t] = _y

        return out.cuda()

    def _predict(self, g, y):
        yl = self.dropout(self.embedding(torch.swapaxes(y[0, :].view(-1, 1), 0, 1)))
        h = self.encoder(yl, g).repeat(2, 1, 1)
        out = torch.empty((self.len_seq, y.shape[1], self.n_classes))

        for t in range(0, self.len_seq):
            _y, h = self.decoder(yl, h)
            yl = torch.argmax(_y.squeeze(0), dim=1)
            yl = self.dropout(self.embedding(torch.swapaxes(yl.view(-1, 1), 0, 1)))
            out[t] = _y

        return out.cuda()

    def _emb(self, g, y):
        yl = self.dropout(self.embedding(torch.swapaxes(y[0, :].view(-1, 1), 0, 1)))
        h = self.encoder(yl, g).repeat(2, 1, 1)
        embs = torch.empty((self.len_seq + 1, y.shape[1], self.decoder.dim_hidden_dec))
        embs[0] = torch.mean(h, dim=0)

        for t in range(0, self.len_seq):
            _y, h = self.decoder(yl, h)
            yl = torch.argmax(_y.squeeze(0), dim=1)
            yl = self.dropout(self.embedding(torch.swapaxes(yl.view(-1, 1), 0, 1)))
            embs[t + 1] = torch.mean(h, dim=0)

        return embs

    def fit(self, data_loader, optimizer, criterion):
        train_loss = 0

        self.train()
        for g, labels in data_loader:
            labels = torch.swapaxes(labels, 0, 1).cuda()
            preds = self(g.cuda(), labels)

            optimizer.zero_grad()
            loss = criterion(preds.view(-1, self.n_classes), labels.flatten().cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        return train_loss / len(data_loader)

    def test(self, data_loader, criterion):
        test_loss = 0
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for g, labels in data_loader:
                labels = torch.swapaxes(labels, 0, 1).cuda()
                dummy_labels = torch.zeros((self.len_seq, g.ptr.shape[0] - 1), dtype=torch.long).cuda()
                preds = self._predict(g.cuda(), dummy_labels)
                test_loss += criterion(preds.view(-1, self.n_classes), labels.flatten().cuda()).item()

                list_preds.append(torch.swapaxes(preds, 0, 1))

        return torch.cat(list_preds, dim=0).cpu(), test_loss / len(data_loader)

    def predict(self, data_loader):
        list_preds = list()

        self.eval()
        with torch.no_grad():
            for g, _ in data_loader:
                dummy_labels = torch.zeros((self.len_seq, g.ptr.shape[0] - 1), dtype=torch.long).cuda()
                list_preds.append(torch.swapaxes(self._predict(g.cuda(), dummy_labels), 0, 1))

        return torch.cat(list_preds, dim=0).cpu()

    def emb(self, data_loader):
        list_embs = list()

        self.eval()
        with torch.no_grad():
            for g, _ in data_loader:
                dummy_labels = torch.zeros((self.len_seq, g.ptr.shape[0] - 1), dtype=torch.long).cuda()
                list_embs.append(self._emb(g.cuda(), dummy_labels).cpu())

        return torch.cat(list_embs, dim=1)


def load_emb_net(path_model_file, elem_attrs, dim_hidden_dec, dim_enc, dim_hidden, path_dataset='dataset/tmsr.json'):
    dataset = load_dataset(path_dataset, elem_attrs)
    encoder = Encoder(dataset.dim_elem_feats, dim_enc)
    decoder = Decoder(dataset.n_classes, dim_hidden_dec, n_layers=2, dropout=0.6)
    model = Seq2Seq(encoder, decoder, dim_hidden, len_seq=dataset.max_operations)
    model.load_state_dict(torch.load(path_model_file))

    return model.cuda()


def get_op_embs(dataset, emb_net):
    return emb_net.emb(DataLoader(dataset, batch_size=128, collate_fn=collate))


def pred_to_recipe(preds, fvec, embs, cond_models, data):
    op_types_by_num = {op_types[op]: op for op in op_types.keys()}
    operations = list()
    pred_conds = list()

    for i in range(1, len(preds)):
        if preds[i] == 1:
            break
        else:
            operations.append([op_types_by_num[preds[i]], i])

    if len(operations) != len(data.operations) - 2:
        return None

    for i in range(0, len(operations)):
        op_type = operations[i][0]
        feat_vec = numpy.hstack([fvec.squeeze(0), embs[operations[i][1]]]).reshape(1, -1)
        list_pred_cond = list()

        if op_type == 'MixingOperation':
            list_pred_cond.append(['mixing', data.operations[operations[i][1]].conditions])
        else:
            for cond in op_conds[op_type]:
                task = op_type + '_' + cond
                pred_cond = cond_models[task].predict(feat_vec)[0]

                if isinstance(op_conds[op_type][cond], str):
                    list_pred_cond.append([cond, numpy.exp(pred_cond)])
                else:
                    list_pred_cond.append([cond, op_conds[op_type][cond][int(pred_cond)]])
        pred_conds.append(list_pred_cond)

    query = ''
    for i in range(0, len(operations)):
        if operations[i][0] == 'MixingOperation':
            query += ', {} {}'.format(pred_conds[i][0][0], pred_conds[i][0][1])
        elif operations[i][0] == 'HeatingOperation':
            if len(pred_conds[i]) == 1:
                query += ', heating under {} K'.format(int(pred_conds[i][0][1]))
            else:
                query += ', heating under {} K during {} h'.format(int(pred_conds[i][0][1]), int(pred_conds[i][1][1]))
        elif operations[i][0] == 'CoolingOperation':
            query += ', cooling {}ly'.format(pred_conds[i][0][1])
        elif operations[i][0] == 'SinteringOperation':
            query += ', sintering with {} method'.format(pred_conds[i][0][1])
        elif operations[i][0] == 'PressingOperation':
            query += ', pressing {} MPa'.format(int(pred_conds[i][0][1]))

    return query
