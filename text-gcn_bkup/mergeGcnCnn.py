import numpy as np 
from utils import load_data_cnn

# (19826, 300) (4957, 300) (24783, 300)
cnnRepre = np.load("sen_vecs2.npy")
gcnRepre = np.load("sen_vecs.npy")

print(cnnRepre.shape)
print(gcnRepre.shape)
# print(cnnRepre.mean(axis = 0), cnnRepre.std(axis = 0))
# print(gcnRepre.mean(axis = 0), gcnRepre.std(axis = 0))

repres = np.concatenate([cnnRepre, gcnRepre], axis = 1)
print(repres.shape)
_, train_y, _, test_y = load_data_cnn()


import tensorflow as tf
train_x = repres[:19826, :] 
test_x = repres[19826:, :] 

x = tf.placeholder([None, 500], dtype=tf.float32)
y = tf.placeholder(tf.int32, [None], name="y")




