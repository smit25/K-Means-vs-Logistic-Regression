import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
 #KMEANS CLUSTERING
mushroom=pd.read_csv('/home/smitu/Downloads/mushrooms.csv')
mushroom.head()
#print(data)
data=mushroom.drop(['class'],axis=1)
label=mushroom['class']
for col in data.columns:
	if len(data[col].value_counts())==2:
		encode=LabelEncoder()
		data[col]=encode.fit_transform(data[col])
data=pd.get_dummies(data)
data.head()
encode=LabelEncoder()
label=encode.fit_transform(label)
#0- edible, 1- poison


train_data,test_data,train_label,test_label = train_test_split(data,label,train_size=0.85)
train_data=np.array(train_data)
test_data=np.array(test_data)
#train_data=train_data.reshape(-1,1)
#test_data=test_data.reshape(-1,1)
train_label=np.array(train_label)

#rememeber that KMEANS is unsupervised
model=KMeans(n_clusters=2,init ='k-means++',n_init=20, verbose=0)
model.fit(train_data)
pred=model.predict(test_data)

opp_pred=[]
for i in pred:
	opp_pred.append(pred[i]^1)

opp_pred=np.array(opp_pred)

count_1=0
count_2=0
for i in range(0,len(test_label)):
	if pred[i]==test_label[i]:
		count_1+=1
	if opp_pred[i]==test_label[i]:
		count_2+=1

count=max(count_1,count_2)
print(len(test_data))
print(len(train_data))
print(len(test_label))
print(len(train_label))
print("Accuracy with K-means :" , count*100/len(test_data))
 #LOGISTRIC REGRESSION

model2=LogisticRegression()

model2.fit(train_data,train_label)
pred2=model2.predict(test_data)
count_r=0
for i in range(0,len(test_label)):
	if pred2[i]==test_label[i]:
		count_r+=1
print("Accuracy with Logistic Regression:" , count_r*100/len(test_label))

"""
OUTPUT:
1219
6905
1219
6905
Accuracy with K-means : 88.9253486464315
/home/smitu/.local/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
Accuracy with Logistic Regression: 100.0
"""