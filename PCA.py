import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

df = pd.read_csv(filepath_or_buffer="beers.csv")
df.columns = ['№', 'abv', 'ibu', 'id', 'name', 'style', 'brewery_id', 'ounces']
df.dropna(how='all', inplace=True)

X = df.ix[:].drop(['№', 'style', 'id', 'name'], axis=1).values
y = df.ix[:, 5].values

label_dict = set(y)

feature_dict = {0: 'abv',
                1: 'ibu',
                2: 'brewery_id',
                3: 'ounces'}

n_features = len(feature_dict)

X[np.isnan(X)] = 0

X_std = StandardScaler().fit_transform(X)


def main_sklearn():
    sklearn_pca = sklearnPCA(n_components=3)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    print(Y_sklearn)

    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        for lab in label_dict:
            ax.scatter(Y_sklearn[y == lab, 0],
                        Y_sklearn[y == lab, 1],
                        Y_sklearn[y == lab, 2],
                        label=lab)
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.zlabel('Principal Component 3')
        # plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


def main():

    # with plt.style.context('seaborn-whitegrid'):
    #     plt.figure(figsize=(8, 8))
    #     for cnt in range(n_features):
    #         plt.subplot(2, 2, cnt + 1)
    #         for lab in label_dict:
    #
    #             if type(lab) == float:
    #                 lab = 'nan'
    #
    #             xlocal_b = X[y == lab, cnt]
    #             if len(xlocal_b) == 0:
    #                 continue
    #
    #             plt.hist(xlocal_b,
    #                      label=lab,
    #                      bins=100,
    #                      alpha=0.3,)
    #         plt.xlabel(feature_dict[cnt])
    #     # plt.legend(loc='upper right', fancybox=True, fontsize=8)
    #     # plt.legend(loc='best')
    #
    #     plt.tight_layout()
    #     plt.show()

    # print(X_std.shape[0])
    # print(X_std)
    mean_vec = np.mean(X_std, axis=0)
    # print(mean_vec)

    cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
    print('Covariance matrix \n%s' % cov_mat)
    # print('NumPy covariance matrix: \n%s' % np.cov(X_std.T))

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)

    # cor_mat2 = np.corrcoef(X.T)
    # eig_vals, eig_vecs = np.linalg.eig(cor_mat2)
    #
    # print('Eigenvectors \n%s' % eig_vecs)
    # print('\nEigenvalues \n%s' % eig_vals)

    # u, s, v = np.linalg.svd(X_std.T)
    # print('Vectors U:\n', u)

    for ev in eig_vecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Everything ok!')

    # Make a list of (eigenvalue, eigenvector) tuples# Make a
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Visually confirm that the list is correctly sorted by decreasing eigenvalues
    print('Eigenvalues in descending order:')
    for i in eig_pairs:
        print(i)

    tot = sum(eig_vals)
    var_exp = [(i[0] / tot) * 100 for i in eig_pairs]
    cum_var_exp = np.cumsum(var_exp)

    # with plt.style.context('seaborn-whitegrid'):
    #     plt.figure(figsize=(6, 4))
    #
    #     plt.bar(range(4), var_exp, alpha=0.5, align='center',
    #             label='individual explained variance')
    #     plt.step(range(4), cum_var_exp, where='mid',
    #              label='cumulative explained variance')
    #     plt.ylabel('Explained variance ratio')
    #     plt.xlabel('Principal components')
    #     plt.legend(loc='best')
    #     plt.tight_layout()
    #     plt.show()

    matrix_w = np.hstack((eig_pairs[0][1].reshape(4, 1),
                          eig_pairs[1][1].reshape(4, 1),
                          eig_pairs[2][1].reshape(4, 1))
                         )

    print('Matrix W:\n', matrix_w)

    Y = X_std.dot(matrix_w)

    with plt.style.context('seaborn-whitegrid'):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        for lab in label_dict:
            ax.scatter(Y[y == lab, 0],
                       Y[y == lab, 1],
                       Y[y == lab, 2],
                        label=lab)
        # plt.xlabel('Principal Component 1')
        # plt.ylabel('Principal Component 2')
        # plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main_sklearn()
    main()