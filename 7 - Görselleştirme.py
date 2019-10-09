import numpy as np
from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns

def visualize_2d(X,y,algorithm="tsne",title="Data in 2D",figsize=(9,9)):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    if algorithm=="tsne":
        reducer = TSNE(n_components=2,n_iter=400,angle=0.6)
    elif algorithm=="pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")
    if X.shape[1]>2:
        X = reducer.fit_transform(X)
    else:
        if type(X)==pd.DataFrame:
        	X=X.values
    f, (ax1) = plt.subplots(nrows=1, ncols=1,figsize=figsize)
    sns.scatterplot(X[:,0],X[:,1],hue=y,ax=ax1);
    ax1.set_title(title);
    plt.show();
    
visualize_2d(x,y,algorithm="pca")

visualize_2d(x,y,algorithm="tsne")