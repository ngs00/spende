import pandas
import torch
import joblib
import openai
import time
from torch.utils.data import DataLoader
from util.chem import load_elem_attrs
from util.data import load_dataset, get_k_folds, collate
from util.model import load_emb_net, pred_to_recipe
from util.operation import op_conds
from util.cond import valid_ops


openai.api_key = OPENAI_API_KEY


n_folds = 5
dim_enc = 128
dim_hidden_dec = 128
dim_hidden = 128
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset = load_dataset('dataset/tmsr.json', elem_attrs)
k_folds = get_k_folds(dataset, n_folds, random_seed=0)
tasks = [op + '_' + cond for op in valid_ops for cond in op_conds[op]]


for k in range(0, n_folds):
    dataset_train = k_folds[k][0]
    dataset_test = k_folds[k][1]
    loader_test = DataLoader(dataset_test, batch_size=128, collate_fn=collate)
    model = load_emb_net('save/model_seq2seq_{}.pt'.format(k), elem_attrs, dim_hidden_dec, dim_enc, dim_hidden)
    cond_models = dict()
    syn_recipes = list()

    for task in tasks:
        cond_models[task] = joblib.load('save/model_eng_cond_{}_{}.joblib'.format(task, k))

    preds_test = model.predict(loader_test)
    embs_test = model.emb(loader_test)

    for i in range(0, len(dataset_test.data)):
        product = dataset_test.data[i].products[0].formula
        doi = dataset_test.data[i].doi
        pred_ops = torch.argmax(preds_test[i], dim=1).tolist()
        fvec = dataset_test.data[i].products[0].fvec
        query_op = pred_to_recipe(pred_ops, fvec, embs_test[:, i], cond_models, dataset_test.data[i])

        if query_op is None:
            continue

        query = 'Generate a paragraph to describe a synthesis recipe of {} via the following operations{}.'\
            .format(product, query_op)
        response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{'role': 'user', 'content': query}])
        answer = response['choices'][0]['message']['content']
        syn_recipes.append([product, doi, answer])

        print(k, i, product, doi)
        print(query_op)
        print(answer)
        print('------------------------------------')
        pandas.DataFrame(syn_recipes).to_excel('save/pred_syn_recipes_{}.xlsx'.format(k), index=False, header=False)
        time.sleep(20)
