from sklearn.metrics import confusion_matrix
import numpy as np
a = np.load("best.npy")
b = np.load("best2.npy")
c = np.load("best3.npy")
d = np.load("best4.npy")
print(confusion_matrix(a, b))
print(confusion_matrix(a, c))
print(confusion_matrix(a, d))
print(confusion_matrix(c, d))
