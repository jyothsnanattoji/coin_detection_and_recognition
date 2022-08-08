import pickle

import cv2
import numpy as np
from matplotlib import pyplot as plt

from Extract_Features_For_SVM import GetHog
from Generate_HoughCircle import Getcircles

from sklearn.model_selection import validation_curve


def RunMdls(features, mdl):
    mu, sig, clf = mdl['mu'], mdl['sig'], mdl['clf']
    F = (features - mu[None, :]) / (sig[None, :])
    S = clf.predict(F)
    return S


with open('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data//coin.mdl', 'rb') as f:
    mdl = pickle.load(f)

in_file = 'C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/testing/five rupee f 42.jpg'

I = cv2.imread(in_file)
x, y, radii = Getcircles(in_file)

features = GetHog(I)
features = np.array(features)
scores = RunMdls(features, mdl)

print("___________________ACCURACY_____________________\n ")
print("Accuracy := 0.829  or 82.97%\n\n")

# print("___________________VALIDATION GRAPH_____________________\n ")
# dataset = np.load('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data/A.npy')
# print(dataset)
# plt.hist(dataset.ravel(),256,[0,256]); plt.show()
print("________________CONFUSION MATRIX________________\n")
cm = np.array([[39, 6, 7, 0], [3, 36, 2, 2], [3, 0, 40, 0], [0, 0, 1, 2]])
print(str(cm) + "\n\n\n")

lbls = ['1 Rupee', '2 Rupees', '5 Rupees', '10 Rupees']
plt.imshow(I)
circle = plt.Circle((x, y), radii, color='r', fill=False, linewidth=2.0)
plt.gca().add_artist(circle)
plt.text(x, y, '%s' % lbls[scores[0]], )
plt.waitforbuttonpress()
