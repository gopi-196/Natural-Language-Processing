
from sklearn.model_selection import train_test_split
import torch
import torchtext
from torchtext import data
import torch.optim as optim
import argparse
import os
import pandas as pd

glove = torchtext.vocab.GloVe(name="6B",dim=100)

data_path = "data"
split = 'data'
df = pd.read_csv(os.path.join(data_path, f"{split}.tsv"), sep="\t")

y = df.label
X = df.text
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,test_size=0.2,stratify=y_train)

print('Training Set')
print(y_train.value_counts())
print('\n')
print('Validation Set')
print(y_val.value_counts())
print('\n')
print('Test Set')
print(y_test.value_counts())
print('\n')

train_df =pd.concat([X_train,y_train],axis=1)
val_df =pd.concat([X_val,y_val],axis=1)
test_df =pd.concat([X_test,y_test],axis=1)

lab_df = train_df[train_df['label'] == 0]
lab1_df = train_df[train_df['label'] == 1]
lab_c = lab_df.iloc[0:25]
lab1_c = lab1_df.iloc[0:25]
overfit_df = pd.concat([lab_c,lab1_c])
overfit_df.head()

df1 =pd.merge(X_train,X_test,how='outer')
df1 =pd.merge(df1,X_val,how = 'outer')
print('The total number of unique entries:', X.nunique())
print('The number of unique values in all three splits: ',df1['text'].nunique())

train_df.to_csv("data/train.tsv", sep="\t", index=False)
val_df.to_csv("data/validation.tsv", sep="\t", index=False)
test_df.to_csv("data/test.tsv", sep="\t", index=False)
overfit_df.to_csv("data/overfit.tsv", sep="\t", index=False)


