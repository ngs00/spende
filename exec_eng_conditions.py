import joblib
import numpy
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, mean_absolute_error, r2_score
from util.chem import load_elem_attrs
from util.data import load_dataset, get_k_folds
from util.cond import valid_ops, get_cond_dataset
from util.model import load_emb_net
from util.operation import op_conds


n_folds = 5
dim_enc = 128
dim_hidden_dec = 128
dim_hidden = 128
elem_attrs = load_elem_attrs('res/matscholar-embedding.json')
dataset = load_dataset('dataset/tmsr.json', elem_attrs)
k_folds = get_k_folds(dataset, n_folds, random_seed=0)
tasks = [op + '_' + cond for op in valid_ops for cond in op_conds[op]]
eval_results = {task: [list(), list(), 'prob_type'] for task in tasks}


for k in range(0, n_folds):
    emb_net = load_emb_net('save/model_seq2seq_{}.pt'.format(k), elem_attrs, dim_hidden_dec, dim_enc, dim_hidden)
    datasets_train = get_cond_dataset(k_folds[k][0], emb_net)
    datasets_test = get_cond_dataset(k_folds[k][1], emb_net)

    for task in tasks:
        prob_type = datasets_train[task][1]
        dataset_cond_train = datasets_train[task][0]
        dataset_cond_test = datasets_test[task][0]
        targets_test = dataset_cond_test[:, -1]
        eval_results[task][2] = prob_type

        if prob_type == 'bin_clf':
            model = XGBClassifier(max_depth=5, n_estimators=600, objective='binary:logistic', eval_metric='mlogloss')
        elif prob_type == 'multi_clf':
            model = XGBClassifier(max_depth=5, n_estimators=600, objective='multi:softprob', eval_metric='mlogloss')
        else:
            model = XGBRegressor(max_depth=5, n_estimators=600, eval_metric='mae')
        model = model.fit(dataset_cond_train[:, :-1], dataset_cond_train[:, -1])
        preds_test = model.predict(dataset_cond_test[:, :-1])

        if prob_type == 'bin_clf':
            eval_results[task][0].append(f1_score(targets_test, preds_test))
            eval_results[task][1].append(numpy.sum(targets_test == preds_test) / targets_test.shape[0])
        elif prob_type == 'multi_clf':
            eval_results[task][0].append(f1_score(targets_test, preds_test, average='macro'))
            eval_results[task][1].append(numpy.sum(targets_test == preds_test) / targets_test.shape[0])
        else:
            eval_results[task][0].append(r2_score(targets_test, preds_test))
            eval_results[task][1].append(mean_absolute_error(targets_test, preds_test))

        joblib.dump(model, 'save/model_eng_cond_{}_{}.joblib'.format(task, k))


for task in tasks:
    print('{} (Problem type: {})'.format(task, eval_results[task][2]))
    print(numpy.mean(eval_results[task][0]), numpy.std(eval_results[task][0]))
    print(numpy.mean(eval_results[task][1]), numpy.std(eval_results[task][1]))
    print('-------------------------------------------------------------')
