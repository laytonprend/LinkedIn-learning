# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 12:29:41 2022

@author: layto
"""

##pca
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
#Kmeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
#score
from sklearn.metrics import accuracy_score
#random forest

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Regression Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor

# Classifer Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

# To visualize individual decision trees
from sklearn import tree
from sklearn.tree import export_text
#other
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

data=pd.read_csv('wine dataset.csv')
print(data.head())
print(data.columns)
#print(data['Wine'].unique())

##pca
#columns to run in PCA could be automate but cbs


features = ['Alcohol','Flavanoids','Proline']
#features = data.columns#may need to drop columns
data['target']=data['Wine']
x = data.loc[:, features].values
y = data.loc[:,['target']].values
x = StandardScaler().fit_transform(x)

#make 2d
# Make an instance of PCA
pca = PCA(n_components=2)

# Fit and transform the data
principalComponents = pca.fit_transform(x)

principalData = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
#Visualise in 2d
finalData = pd.concat([principalData, data[['target']]], axis = 1)
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (8,8));
targets = data.loc[:, 'target'].unique()
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalData['target'] == target
    ax.scatter(finalData.loc[indicesToKeep, 'Principal Component 1']
               , finalData.loc[indicesToKeep, 'Principal Component 2']
               , c = color
               , s = 50)

ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)    
ax.legend(targets)
ax.grid()
plt.show()

#explain variance
print('PCA what variance explained in each variable',pca.explained_variance_ratio_)
print('PCA what variance explained',sum(pca.explained_variance_ratio_))#0.9 accounted for
#print(finalData.head())

#now do KMeansclustering

#TrainTestSplit
x=finalData[['Principal Component 1', 'Principal Component 2']]
X_train, X_test, y_train, y_test = train_test_split(x, finalData['target'], random_state=0)


#data = load_wine()
#wine = pd.DataFrame(data.data, columns=data.feature_names)
#print(finalData.head()) 
#X_train = finalData[['Principal Component 1', 'Principal Component 2']] 

#scale = StandardScaler()
#scale.fit(X_train)
#X_train_scaled = scale.transform(X_train)
#print('X_train\n',X_train)
#print('X_train[0]\n',X_train[0])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

plt.scatter(X_test['Principal Component 1'],#X_train[:,0], 
            X_test['Principal Component 2'],
            c= y_pred)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], 
            marker="*",
            s = 250, 
            c = [0,1,2], 
            edgecolors='k')
plt.xlabel('Principal Component 1'); plt.ylabel('Principal Component 2')
plt.title('k-means (k=3)')
plt.show()

#elbow method
import numpy as np
# calculate distortion for a range of number of cluster
inertia = []
for i in np.arange(1, 11):
    km = KMeans(
        n_clusters=i
    )
    km.fit(X_train)
    inertia.append(km.inertia_)
#print(y_test,y_pred)
print('Kmeans',accuracy_score(y_test, y_pred))#75% accurate at grouping123b.
# plot
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()




#accurate clustering,
#now do logistic regression on whether == wine 2, lamda needed

def myLogisticRegression(x,finalData,TargetCategory):
    
#, 'Principal Component 2']]
    finalData['target==TargetCategory']=finalData.apply(lambda x: 1 if x.target==TargetCategory else 0,axis=1)
    #ReviewData.apply(lambda x: 1 if x.Health_Inspector_Data == True else 0, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(x, finalData['target==TargetCategory'], random_state=0)
   # print('logistic regression\n',X_train.head())
    #restandardise
    '''scaler = StandardScaler()
    
    # Fit on training set only.
    scaler.fit(X_train)
    
    # Apply transform to both the training set and the test set.
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)'''
    ##x = data.loc[:, features].values
    #y = data.loc[:,['target']].values
    #x = StandardScaler().fit_transform(x)
    #scaler = StandardScaler()
    
    # Fit on training set only.
    #scaler.fit(X_train)
    
    # Apply transform to both the training set and the test set.
    
    X_train = StandardScaler().fit_transform(X_train)#scaler.transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)#caler.transform(X_test)
    
    
    clf = LogisticRegression()
    clf.fit(X_train,y_train)
   # print('prediction', clf.predict(X_test[0].reshape(1,-1))[0])
    #print('probability', clf.predict_proba(X_test[0].reshape(1,-1)))
    
    Prediction=clf.predict(X_test)
    #print('pred',Prediction)
    #print(accuracy_score(y_test, Prediction))
    return y_test, Prediction
Totaly_test, TotalPrediction=myLogisticRegression(x,finalData,1)

y_test, Prediction=myLogisticRegression(x,finalData,2)
#Totaly_test=Totaly_test+y_test
#TotalPrediction=TotalPrediction+Prediction


Totaly_test=[*Totaly_test,*y_test]
TotalPrediction=[*TotalPrediction,*Prediction]
#inear_y_pred_array=[*Linear_y_pred,*Linear_y_pred_array]#need to union these arrays to see overall value
 #                   Linear_Y_test_array=[*Linear_Y_test,*Linear_Y_test_array]

#print(len(TotalPrediction))
y_test, Prediction=myLogisticRegression(x,finalData,3)
Totaly_test=[*Totaly_test,*y_test]
TotalPrediction=[*TotalPrediction,*Prediction]
#print(len(TotalPrediction))
print('Logistic Regresion',accuracy_score(Totaly_test, TotalPrediction))


##now make random forest
def myRandomForest(x,y,target):
    X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)
    X_train = StandardScaler().fit_transform(X_train)#scaler.transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)#caler.transform(X_test)
   # print(X_test)
    reg = RandomForestRegressor(n_estimators=10, random_state = 0)
    reg.fit(X_train, y_train)
    y_pred=reg.predict(X_test)#.iloc[0].values.reshape(1, -1))
    #reg.predict(X_test)
   # print(y_test,y_pred)
    #print(accuracy_score(y_test, y_pred))
    
    #show
    #rewrite, filter y_pred and y_test based on what actual value is 
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10,7));
    #df=X_test
    df = pd.DataFrame(X_test, columns = ['Principal Component 1','Principal Component 2'])
    
    df['logistic_preds']=y_pred
    df['target']=y_test
    #print('Random Forest',accuracy_score(df['target'], df['logistic_preds']))
    
    #X_test['logistic_preds']=y_pred
    #df=X_test
   # df['logistic_preds']=y_pred
    targetFilter = df['target'] == target
    nottargetFilter = df['target'] !=target
    dfTarget=df[targetFilter]
    dfNotTarget=df[nottargetFilter]
    
    ax.scatter(dfTarget['Principal Component 1'],
               dfTarget['Principal Component 2'],#df.loc[targetFilter, 'logistic_preds'],
               color = 'g',
               s = 60,
               label = 'Target')
    
    
    ax.scatter(dfNotTarget['Principal Component 1'],
               dfNotTarget['Principal Component 2'],
               color = 'b',
               s = 60,
               label = 'NotTarget')
    
    ax.axhline(y = .5, c = 'y')
    
    ax.axhspan(.5, 1, alpha=0.05, color='green')
    ax.axhspan(0, .4999, alpha=0.05, color='blue')
    ax.text(0.5, .6, 'Classified as target', fontsize = 16)
    ax.text(0.5, .4, 'Classified as not target', fontsize = 16)
    
    ax.set_ylim(0,1)
    ax.legend(loc = 'lower right', markerscale = 1.0, fontsize = 12)
    ax.tick_params(labelsize = 18)
    ax.set_xlabel('?', fontsize = 24)#neds fixing
    ax.set_ylabel('probability of target', fontsize = 24)
    ax.set_title('Logistic Regression Predictions', fontsize = 24)
    fig.tight_layout()
    
    
#x=finalData[['Principal Component 1', 'Principal Component 2']]
#X_train, X_test, y_train, y_test = train_test_split(x, finalData['target'], random_state=0)

print(x.head())
#print(x.type())
myRandomForest(x,finalData['target'],1)#seems to fail,maybe doesn't wokr if integer
#myRandomForest(data[['features']],finalData['target'])   
    

##My First ML Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Train Test Split
#X_train, X_test, y_train, y_test = train_test_split(df[df.columns[:-1]], df['label'], random_state=0)
X_train, X_test, y_train, y_test = train_test_split(x, finalData['target'], random_state=0)
# Create a pipeline
pipe = Pipeline([('scaler', StandardScaler()),
                 ('pca', PCA(n_components = 0.9, random_state=0)),#was 0.9?
                 ('logistic', LogisticRegression())])

pipe.fit(X_train, y_train)

# Get Model Performance
print('Pipe',pipe.score(X_test, y_test))


from sklearn import set_config

set_config(display='diagram')
print(pipe)
pipe
set_config(display='text')






