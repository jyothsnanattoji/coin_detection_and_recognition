import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from Extract_Features_For_SVM import GetHog
from Generate_HoughCircle import Getcircles
import os


def RunMdls(features, mdl):
    mu, sig, clf = mdl['mu'], mdl['sig'], mdl['clf']
    F = (features - mu[None, :]) / (sig[None, :])
    S = clf.predict(F)
    return S


if __name__ == "__main__":
    with open('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data/coin.mdl', 'rb') as f:
        mdl = pickle.load(f)

    in_file = 'E:/ZSTUDY/VIMPORTANT/Coin Recognition And Detection/Coin Data Set Kaggle/testing/five rupee f 2.jpg'

    root = "C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/testing"
    for f in os.listdir(root):
        if f.endswith(".jpg"):
            in_file = os.path.join(root, f)
            print(in_file)
            # if "five" in in_file:
            #     continue

            I = cv2.imread(in_file)
            x, y, radii = Getcircles(in_file)

            features = GetHog(I)
            features = np.array(features)
            scores = RunMdls(features, mdl)

            if "one" in in_file:
                print("----ONE------")
            if "two" in in_file:
                print("----TWO------")
            if "five" in in_file:
                print("----FIVE------")
            if "ten" in in_file:
                print("----TEN------")

            lbls = ['1 Rupee', '2 Rupees', '5 Rupees', '10 Rupees']
            print(lbls[scores[0]])

            # plt.imshow(I)
            # circle = plt.Circle((x, y), radii, color='r', fill=False, linewidth=2.0)
            # plt.gca().add_artist(circle)
            # plt.text(x, y, '%s' % lbls[scores[0]])
            # plt.waitforbuttonpress()
