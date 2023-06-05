from pandas import DataFrame
from util.chem import load_elem_attrs
from util.data import get_k_folds
from util.model import *


n_folds = 5
batch_size = 64
epochs = 1000
dim_enc = 128
dim_hidden_enc = 128
dim_hidden_dec = 128


elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset = load_dataset('dataset/tmsr.json', elem_attrs)
k_folds = get_k_folds(dataset, n_folds, random_seed=0)


for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, collate_fn=collate)

    encoder = Encoder(dataset.dim_elem_feats, dim_enc)
    decoder = Decoder(dataset_train.n_classes, dim_hidden_dec, n_layers=2, dropout=0.6)
    model = Seq2Seq(encoder, decoder, dim_hidden_enc, len_seq=dataset_train.max_operations).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-6)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(0, epochs):
        loss_train = model.fit(loader_train, optimizer, criterion)
        _, loss_test = model.test(loader_test, criterion)
        print('Repeat [{}/{}]\tEpoch [{}/{}]\tTrain Loss {:.4f}\tTest Loss {:.4f}'
              .format(k + 1, n_folds, epoch + 1, epochs, loss_train, loss_test))
    torch.save(model.state_dict(), 'save/model_seq2seq_{}.pt'.format(k))

    pred_results = list()
    preds_test = model.predict(loader_test)
    for i in range(0, len(dataset_test.data)):
        data_info = [dataset_test.idx_data[i], dataset_test.forms[i]]
        pred_results.append(data_info + dataset_test.labels[i].tolist())
        pred_results.append(data_info + torch.argmax(preds_test[i], dim=1).tolist())
    DataFrame(pred_results).to_excel('save/preds_seq2seq_{}.xlsx'.format(k), index=False, header=False)
