import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from sklearn.cluster import KMeans

dataset = pd.read_csv('cluster_Data.csv')
dataset.head()
df = pd.read_csv('cluster_Data.csv')
df.head()



dataset.isnull().sum()

dataset_new = dataset[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].values

limit = int((dataset_new.shape[0] // 2) ** 0.5)


wcss = {}

for k in range(2, limit + 1):
    model = KMeans(n_clusters=k)
    model.fit(dataset_new)
    wcss[k] = model.inertia_

plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel('Values of "k"')
plt.ylabel('WCSS')
plt.show()




model = KMeans(n_clusters=5)

# predicting the clusters
pred = model.fit_predict(dataset_new)

outcome = []
for i in model.labels_:
    outcome.append(i)

df = df.assign(outcome=outcome)

# plotting all the clusters
colours = ['red', 'blue', 'green', 'yellow', 'orange']

for i in np.unique(model.labels_):
    plt.scatter(dataset_new[pred == i, 0],
                dataset_new[pred == i, 1],
                c=colours[i])

# plotting the cluster centroids
plt.scatter(model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            s=200,  # marker size
            c='black')

plt.title('K Means clustering')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Pairplot

colours = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange'}

seaborn.pairplot(df, hue='outcome', palette=colours)

plt.show()

