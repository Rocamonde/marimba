from sklearn.decomposition import PCA


def pca(covariates, n_components):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(covariates), pca
