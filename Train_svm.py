from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import pickle

X, Y = [], []
for i, t in enumerate(['A', 'B', 'C', 'D']):
    # with open('./Features/%s.npy' % t, 'rb') as f:
    with open('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data/%s.npy' % t, 'rb') as f:
        features = np.load(f)
    X += [features]
    Y += [i * np.ones(features.shape[0])]
X = np.concatenate(X).astype(np.float64, 'C')
Y = np.concatenate(Y).astype(np.int32, 'C')

idx = np.random.permutation(Y.size)
X, Y = X[idx, :], Y[idx]

mu = np.mean(X, 0)
X = X - mu[None, :]
sig = np.std(X, 0)
X = X * (1.0 / sig[None, :])

clf = svm.SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
scores = cross_val_score(clf, X, Y, cv=5)

clf.fit(X, Y)

mdl = {'clf': clf, 'mu': mu, 'sig': sig}
with open('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data/coin.mdl', 'wb') as f:
    pickle.dump(mdl, f)
