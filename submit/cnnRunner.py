from colorama import init, Fore
import numpy as np
import os
from sklearn import metrics
from sklearn.utils import shuffle
from utils import make_vocab, load_data_cnn
from wordCNN import *

init() #coloroma init

# embedding_dim = 200
# golveFileName = os.path.join("data", "twitter_hate_off_word_vectors.txt")
# saveFileName = os.path.join("data", "twitter_hate_off_word_vectors" + str(embedding_dim) + ".npy")
embedding_dim = 100
golveFileName = os.path.join("data", "glove.twitter.27B." + str(embedding_dim) + "d.txt")
saveFileName = os.path.join("data", "filteredGlove" + str(embedding_dim) + ".npy")
vocab_size = make_vocab(file = golveFileName, save_name = saveFileName, embedding_dim = embedding_dim)
wordVecs = np.load(saveFileName).astype(np.float32)

train_x, train_y, dev_x, dev_y, test_x, test_y, doc_emb_train, doc_emb_test, doc_emb_dev = load_data_cnn()

n_epochs = 20
train_instances = len(train_x)
batch_size = 128
train_batches = train_instances // batch_size

use_gcn = True
path1 = "./saved/use_gcn_cnn"
path2 = "./saved/no_use_gcn_cnn" 
if use_gcn:
  path = path1
else:
  path = path2

model = WordCNN(vocab_size, 64, 3, wordVecs, embedding_dim, use_gcn)
# model.load_model(path2)

ans = 0
best_ans = []
best_repre = None
all_test_f1_weighted = []

def test_on_data(model, data, doc_emb, orig_y):
  n_batches = len(data) // batch_size

  y_pred = []
  for i in range(n_batches):
    temp = model.test_on_batch(
      data[i * batch_size : (i + 1) * batch_size], 
      doc_emb[i * batch_size : (i + 1) * batch_size])
    y_pred.extend(temp)

  temp = model.test_on_batch(
    data[n_batches * batch_size : ],
    doc_emb[n_batches * batch_size : ])
  y_pred.extend(temp)
  
  print(Fore.RED + "Result data:")
  pred_y = y_pred
  print(Fore.RED + " Precision, Recall and F1-Score : ")
  print(metrics.classification_report(orig_y, pred_y, digits=4))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='macro')
  print(Fore.RED + "Macro average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='micro')
  print(Fore.RED + "Micro average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='weighted')
  print(Fore.RED + "Weight average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))

  return f

for i in range(n_epochs):
  # train_x, train_y = shuffle(train_x, train_y)
  print("Running epoch number : ", i+1)
  for batch_no in range(int(train_batches)):
    if batch_no % 20 == 0:
      print("Batch no : ", batch_no)
    batch_train_x = train_x[batch_no * batch_size : (batch_no + 1) * batch_size]  
    batch_doc_x = doc_emb_train[batch_no * batch_size : (batch_no + 1) * batch_size]  
    batch_train_y = train_y[batch_no * batch_size : (batch_no + 1) * batch_size]
    batch_doc_x = np.array(batch_doc_x)
    if i < 10:
      batch_doc_x *= 0
    model.train_on_batch(batch_train_x, batch_train_y, batch_doc_x)
    if i == 10 and i == train_batches - 1 and use_gcn:
      print("Loaded saved model")
      model.load_model(path)
    


  print(Fore.RED + "Result on train data:")
  test_on_data(model, train_x, doc_emb_train, train_y)

  print(Fore.GREEN + "Result on dev data:")
  f = test_on_data(model, dev_x, doc_emb_dev, dev_y)
  if f > ans:
    ans = f
    model.save_model(path)

  print(Fore.BLUE + "Result on test data:")
  f1 = test_on_data(model, test_x, doc_emb_test, test_y)
  all_test_f1_weighted.append((f, f1))

model.load_model(path)
print(Fore.BLUE + "Result on test data:")
f = test_on_data(model, test_x, doc_emb_test, test_y)


print("\n\n F1 weighted test score : ", f)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(all_test_f1_weighted)
plt.savefig("weighted_f1_test.png")

np.save("best5.npy", best_ans)
np.save("testy.npy", test_y)
for op in all_test_f1_weighted:
  print(op)

# np.save("sen_vecs2.npy", best_repre)
