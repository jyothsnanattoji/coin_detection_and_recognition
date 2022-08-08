import numpy as np
from skimage.feature import hog
from scipy.ndimage import rotate
import cv2
import os


# HOG Function to Extract Features For SVM
def GetHog(patch):
    patch = cv2.resize(patch, (64, 64))

    fd, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True, multichannel=True)

    return fd


# Extracting Features for each individual coin And storing it as npy(numpy array extension)
if __name__ == '__main__':
    for t in ['A', 'B', 'C', 'D']:
        skip = 60
        root = "C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/training/{}".format(t)
        # root = './Training_Set/%s/' % t
        features = []
        for f in os.listdir(root):
            if f.endswith(".jpg") or f.endswith(".JPG"):
                in_file = os.path.join(root, f)
                print(in_file)

                I = cv2.imread(in_file)
                for roll in range(0, 360, skip):
                    Ir = rotate(I, roll, reshape=False)
                    F = GetHog(Ir)
                    features += [F]

        features = np.array(features, dtype=np.float)
        with open('C:/Users/Jyo/Desktop/Ml/Latest code and Dataset/Coin Data Set Kaggle/train_data/%s.npy' % t, 'wb') as extracted_data:
            np.save(extracted_data, features)

