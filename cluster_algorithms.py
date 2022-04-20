import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize



def Calinski_Harabasz(df_in):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 30), metric='calinski_harabasz', timings=True)
    visualizer.fit(df_in)
    return visualizer.show()

def Elbow_Method(df_in):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 30), timings=True)
    visualizer.fit(df_in)  # Fit data to visualizer
    return visualizer.show()


def silhouette_score(df_in):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(2, 30), metric='silhouette', timings=True)
    visualizer.fit(df_in)  # Fit the data to the visualizer
    return visualizer.show()




def PCA_analys(df_in):
    sc = StandardScaler()
    df_in = sc.fit_transform(df_in)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_in)
    pca_df = pd.DataFrame(columns=['pca1', 'pca2'])
    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))
    return pca_df



def TSNE_analys(df_in):
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(df_in)
    tsne_df = pd.DataFrame(columns=['tsne1', 'tsne2'])
    tsne_df['tsne1'] = tsne_result[:, 0]
    tsne_df['tsne2'] = tsne_result[:, 1]
    return tsne_df




def plot_clustering(df_in, n, title):

    df_in = df_in.values

    model = KMeans(n_clusters=n)
    model.fit(df_in)

    pred = model.fit_predict(df_in)

    labels = []
    for i in model.labels_:
        labels.append(i)

    colours = ['red', 'blue', 'green', 'yellow', 'orange']

    for i in np.unique(model.labels_):
        plt.scatter(df_in[pred == i, 0],
                    df_in[pred == i, 1],
                    c=colours[i])

    plt.scatter(model.cluster_centers_[:, 0],
                model.cluster_centers_[:, 1],
                s=200,  # marker size
                c='black')

    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    return plt.show()


def GaussianMixture_Bic(df_in):
    n_components = np.arange(1, 21)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(df_in)
              for n in n_components]

    plt.plot(n_components, [m.bic(df_in) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    return plt.show()


def Dbscan(df_in):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_in)

    x_normal = normalize(x_scaled)
    x_normal = pd.DataFrame(x_normal)

    pca = PCA(n_components=2)
    x_principal = pca.fit_transform(x_normal)
    x_principal = pd.DataFrame(x_principal)
    x_principal.columns = ['V1', 'V2']

    dbscan = DBSCAN(eps=0.2, min_samples=4).fit(x_principal)
    labels = dbscan.labels_
    df_in['cluster'] = dbscan.labels_

    clusterColor = {0: u'yellow', 1: u'green', 2: 'blue', -1: u'red', 3: 'black', 4: 'orange'}
    colors = [clusterColor[label] for label in labels]
    plt.figure(figsize=(12, 10))
    plt.scatter(x_principal['V1'], x_principal['V2'], c=colors)
    plt.title("Implementation of DBSCAN Clustering", fontname="Times New Roman", fontweight="bold")
    return plt.show()


filepath = 'cluster_Data.csv'
dataset = pd.read_csv(filepath)

ch_score = Calinski_Harabasz(dataset)
sc = silhouette_score(dataset)
elbow_score = Elbow_Method(dataset)





pca_df = PCA_analys(dataset)
pca_df = plot_clustering(pca_df, n=4, title='PCA-Clustering')



tsne_df = TSNE_analys(dataset)
tsne_df = plot_clustering(tsne_df, n=4, title='tsne-Clustering')


bic = GaussianMixture_Bic(dataset)


ds = Dbscan(dataset)







