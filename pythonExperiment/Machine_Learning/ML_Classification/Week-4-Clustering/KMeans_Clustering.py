import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

print(dataset)

X = dataset.iloc[:, 3:5].values

print(X)

from sklearn.cluster import KMeans
list1 = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    list1.append(kmeans.inertia_)
plt.plot(range(1, 11), list1)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


print(list1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


print(y_kmeans)

supervised=dataset
supervised['Cluster_group']=y_kmeans

print(supervised)

supervised.to_csv("cluster.csv",index=False)

dir(kmeans)

centroids=kmeans.cluster_centers_

centroids


print(centroids)
y_kmeans

#import seaborn as sns
#facet = sns.lmplot(data=supervised, x=supervised.columns[3], y=supervised.columns[4], hue=supervised.columns[5],
                   #fit_reg=False,legend_out=True)

import seaborn as sns
facet = sns.lmplot(data=supervised, x=supervised.columns[3], y=supervised.columns[4],
                   fit_reg=False)

print(facetcd )