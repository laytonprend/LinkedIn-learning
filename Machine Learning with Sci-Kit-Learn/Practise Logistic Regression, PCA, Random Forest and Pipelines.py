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
print(data['Wine'].unique())

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

#explain variance
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))#0.9 accounted for
print(finalData.head())

#now do KMeansclustering

#TrainTestSplit
X_train, X_test, y_train, y_test = train_test_split(data['Principal Component 1', 'Principal Component 2'], data['target'], random_state=0)


#data = load_wine()
#wine = pd.DataFrame(data.data, columns=data.feature_names)
print(finalData.head()) 
X = finalData[['Principal Component 1', 'Principal Component 2']] 

#scale = StandardScaler()
#scale.fit(X)
#X_scaled = scale.transform(X)
#print('X\n',X)
#print('X[0]\n',X[0])
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X['Principal Component 1'],#X[:,0], 
            X['Principal Component 2'],
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
    km.fit(X)
    inertia.append(km.inertia_)
print(kmeans.score(y_test,y_pred))
# plot
plt.plot(np.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()




#so wine is a clustering already done, we do our clustering ignoring this but compare to after


#X_train, X_test, y_train, y_test = train_test_split(data
#data=StandardScaler.fit(data)
#print(data.head())