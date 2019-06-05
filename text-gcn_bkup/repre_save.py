import numpy as np
with open("data/twitter.train.index", "r") as f:
  train_indices = [int(line.strip()) for line in f.readlines()]

with open("data/twitter.test.index", "r") as f:
  test_indices = [int(line.strip()) for line in f.readlines()]

sen_vecs = np.zeros((len(train_indices) + len(test_indices), 200))

with open("data/twitter_doc_vectors.txt", "r") as f:
  for line in f:
    temp = line.split()
    line_no = int(temp[0].split('_')[1])
    vec = [float(k) for k in temp[1:]]
    sen_vecs[line_no] = vec

np.save("sen_vecs.npy", sen_vecs)
