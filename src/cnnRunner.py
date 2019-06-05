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
vocab_size = make_vocab(data = "twitter_hate_off", file = golveFileName, save_name = saveFileName, embedding_dim = embedding_dim)
print(vocab_size)
wordVecs = np.load(saveFileName).astype(np.float32)

train_x, train_y, dev_x, dev_y, test_x, test_y, doc_emb_train, doc_emb_test, doc_emb_dev = load_data_cnn(data = "twitter_hate_off")

n_epochs = 5
train_instances = len(train_x)
batch_size = 128
train_batches = train_instances // batch_size

use_gcn = False
path1 = "./saved/use_gcn_cnn"
path2 = "./saved/no_use_gcn_cnn" 
if use_gcn:
  path = path1
else:
  path = path2

model = WordCNN(vocab_size, 64, 3, None, embedding_dim, use_gcn)
# model.load_model(path2)

ans = 0
best_ans = []
best_repre = None
all_test_f1_weighted = []

def test_on_data(model, data, doc_emb, orig_y, color):
  n_batches = len(data) // batch_size

  y_pred = []
  embeds = []
  for i in range(n_batches):
    temp = model.test_on_batch(
      data[i * batch_size : (i + 1) * batch_size], 
      doc_emb[i * batch_size : (i + 1) * batch_size])
    y_pred.extend(temp[0])
    embeds.extend(temp[1])

  temp = model.test_on_batch(
    data[n_batches * batch_size : ],
    doc_emb[n_batches * batch_size : ])
  y_pred.extend(temp[0])
  embeds.extend(temp[1])
  
  print(color + "Result data:")
  pred_y = y_pred
  print(color + " Precision, Recall and F1-Score : ")
  print(metrics.classification_report(orig_y, pred_y, digits=4))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='macro')
  print(color + "Macro average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='micro')
  print(color + "Micro average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))
  p, r, f, s = metrics.precision_recall_fscore_support(orig_y, pred_y, average='weighted')
  print(color + "Weight average Precision, Recall and F1-Score : {:.2f} {:.2f} {:.2f}".format(p, r, f))

  return f, embeds

temp = 0
best_embeds = ()
for i in range(n_epochs):
  print(Fore.WHITE + "Running epoch number : {}".format(i+1))
  for batch_no in range(int(train_batches)):
    if batch_no % 20 == 0:
      print("Epoch , Batch no : {} {}".format(i, batch_no))

      print(Fore.GREEN + "Result on dev data:")
      f, embed_dev = test_on_data(model, dev_x, doc_emb_dev, dev_y, Fore.GREEN)
      print(Fore.RED + "Result on train data:")
      _, embed_train =test_on_data(model, train_x, doc_emb_train, train_y, Fore.RED)

      print(Fore.BLUE + "Result on test data:")
      f1, embed_test = test_on_data(model, test_x, doc_emb_test, test_y, Fore.BLUE)
      all_test_f1_weighted.append((f, f1))
      if f > ans:
        ans = f
        model.save_model(path)
        temp = (f, f1)
        best_embeds = (embed_train, embed_dev, embed_test)
      print(Fore.BLUE + "Results dev test : {} {}".format(temp[0], temp[1]))

      
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

model.load_model(path)
print(Fore.BLUE + "Result on test data:")
f = test_on_data(model, test_x, doc_emb_test, test_y, Fore.BLUE)


print("\n\n F1 weighted test score : {}".format(f))
print(ans)

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.plot(all_test_f1_weighted)
plt.savefig("weighted_f1_test.png")

np.save("best5.npy", best_ans)
np.save("testy.npy", test_y)
for op in all_test_f1_weighted:
  print(op)

# np.save("sen_vecs2.npy", best_repre)
import pickle
with open("saved_everything_cnn", "wb") as f:
    pickle.dump(best_embeds, f)
