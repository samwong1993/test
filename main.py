import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

df = pd.read_excel('./df1.xlsx')
df.fillna(0,inplace=True)
features = ['chat_time', 'bill', 'chat',
       'others', 'bill_money', 'chat_money', 'bill_time', 'business']
df['pred'] = 0
df['pred'].loc[(df.chat >0)& (df.bill>0)] = 1
auc_score = roc_auc_score(df['label'],df['pred'])
print(auc_score)


label = ['label']
for i in features:
    df[i] = df[i].astype('int')

# Soil_ACd = w_1 * Plant + w_2 * Treatment ....
X_train,X_test,y_train,y_test=train_test_split(df[features],df[label],test_size=0.05,random_state=20)
train_dataset = Dataset(X_train, label=y_train, free_raw_data=False)
valid_dataset = Dataset(X_test, label=y_test, free_raw_data=False)

params = {
    'objective': 'binary',
    'boosting': 'rf',
    'metric': 'auc',
    'num_iteration': 10000,
    'learning_rate': 0.01,
    'num_leaves': 8,
    # 'verbose': -1,
    # 'seed': 7,
    'mode': 'sequential_covering',
    'min_data_in_leaf': 5,
    # 'lambda_l2': 0.1,
    # 'lambda_l1': 0.1,
    # 'max_bin': 8,
    'max_depth': 3,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'min_data_in_bin': 1,
    'rule_top_n_each_tree': 1,
    'num_stop_threshold': 10000,
    # 'categorical_feature': '2,5'
}


booster, rules = lgb.train(params, train_dataset)
# booster = lgb.train(params, train_dataset)

dmodel = booster.dump_model()
str_model = booster.model_to_string()

pred = booster.predict(train_dataset.get_data(), pred_leaf=True)
booster.save_model('model.txt')
paths = booster.get_leaf_path(missing_value=True)
dataframe = pd.DataFrame(rules)
dataframe.to_csv('output.csv')
print(pred)
TP = 0
FP = 0
TN = 0
FN = 0
num0 = []
num1 = []
x = []
y_recall = []
y_acc = []
for i in range(len(rules)):
    num0.append(rules[i]['rule_info'][0][0])
    num1.append(rules[i]['rule_info'][0][1])
for i in range(len(rules)-1):
    TP = TP + rules[i]['rule_info'][0][1]
    FP = FP + rules[i]['rule_info'][0][0]
    TN = sum(num0[i+1:])
    FN = sum(num1[i+1:])
    train_precision = TP / (TP + FP + 1e-10)
    train_recall = TP / (TP + FN + 1e-10)
    train_accuracy = (TP + TN)/(TP + FN + FP + TN)
    print(f"Number of rules: {i}")
    print('TP:{} FP:{} TN:{} FN:{}'.format(TP,FP,TN,FN))
    print('Train set: Positive:{} Negative:{}'.format(sum(num1),sum(num0)))
    print('train_precision:{:.2f}% train_recall:{:.2f}% train_accuracy:{:.2f}%'.format(100*train_precision,100*train_recall,100*train_accuracy))
    x.append(i)
    y_acc.append(train_accuracy)
    y_recall.append(train_recall)

import matplotlib.pyplot as plt
plt.plot(x,y_acc,label = 'Accuracy')
plt.plot(x,y_recall,label='Recall')
plt.xlabel('Number of rules')
plt.ylabel('Percentage')
plt.legend(['Accuracy','Recall'])
plt.show()
import os
os.environ["PATH"] += os.pathsep + 'd:/Graphviz/bin'
graph = lgb.create_tree_digraph(booster, tree_index=0)
graph.render(view=True)
print("************************************************************")
TP = df.loc[(df.pred_old == 1) & (df.label == 1)].shape[0]
FP = df.loc[(df.pred_old == 1) & (df.label == 0)].shape[0]
TN = df.loc[(df.pred_old == 0) & (df.label == 0)].shape[0]
FN = df.loc[(df.pred_old == 0) & (df.label == 1)].shape[0]
# TP = df.loc[df.category == "TT"].shape[0]
# FP = df.loc[df.category == "TF"].shape[0]
# TN = df.loc[df.category == "FT"].shape[0]
# FN = df.loc[df.category == "FF"].shape[0]
train_precision = TP / (TP + FP + 1e-10)
train_recall = TP / (TP + FN + 1e-10)
train_accuracy = (TP + TN) / (TP + FN + FP + TN)
print('TP:{} FP:{} TN:{} FN:{}'.format(TP, FP, TN, FN))
print('train_precision:{:.2f}% train_recall:{:.2f}% train_accuracy:{:.2f}%'.format(100 * train_precision,
                                                                                   100 * train_recall,
                                                                                   100 * train_accuracy))









