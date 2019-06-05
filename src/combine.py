from colorama import init, Fore
import tensorflow as tf
import numpy as np
import os
import pickle
from sklearn import metrics
from sklearn.utils import shuffle
from utils import make_vocab, load_data_cnn

init() #coloroma init
_, train_y, _, dev_y, _, test_y, doc_emb_train, doc_emb_test, doc_emb_dev = load_data_cnn(data = "twitter_hate_off")

train_x = doc_emb_train
dev_x = doc_emb_dev
test_x = doc_emb_test

with open("saved_everything_cnn", "rb") as f:
    embed_train, embed_dev, embed_test = pickle.load(f)

train_x = embed_train
dev_x = embed_dev
test_x = embed_test

embedding_dim = 300
learning_rate = 1e-3

x = tf.placeholder(tf.float32, [None, embedding_dim], name="x")
keep_probab = tf.placeholder(tf.float32)
x1 = tf.nn.dropout(x, keep_probab)
y = tf.placeholder(tf.int32, [None], name="y")
h = tf.layers.dense(x, 3)
predictions = tf.argmax(h, -1, output_type=tf.int32)
loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


def test_on_data(data, orig_y, color):
  """
  Function to test
  """
  n_batches = len(data) // batch_size
  y_pred = []
  for i in range(n_batches):
    temp = sess.run([predictions], feed_dict={x: data[i * batch_size: (i + 1) *batch_size], keep_probab: 1.0})
    y_pred.extend(temp[0])
  
  temp = sess.run([predictions], feed_dict = {x : data[n_batches * batch_size : ], keep_probab : 1.0})
  y_pred.extend(temp[0])
  
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

  return f

def train_on_batch(orig_x, orig_y):
  sess.run(optimizer, feed_dict = {x : orig_x, y : orig_y, keep_probab : 0.5})

all_test_f1_weighted = []
n_epochs = 10
train_instances = len(train_x)
batch_size = 128
train_batches = train_instances // batch_size
ans = 0.0
best_op = ()
for i in range(n_epochs):
  print(Fore.WHITE + "Running epoch number : {}".format(i+1))
  for batch_no in range(int(train_batches)):
    if batch_no % 20 == 0:
      print("Epoch , Batch no : {} {}".format(i, batch_no))
      print(Fore.GREEN + "Result on dev data:")
      f = test_on_data(dev_x, dev_y, Fore.GREEN)
      print(Fore.RED + "Result on train data:")
      _ = test_on_data(train_x, train_y, Fore.RED)
      print(Fore.BLUE + "Result on test data:")
      f1 = test_on_data(test_x, test_y, Fore.BLUE)
      all_test_f1_weighted.append((f, f1))
      if f > ans:
        ans = f
        best_op = (ans, f1)
      # print(Fore.BLUE + "Results dev test : {} {}".format(ans, f0))
      all_test_f1_weighted.append((f1, f))
      
    batch_train_x = train_x[batch_no * batch_size : (batch_no + 1) * batch_size]  
    batch_train_y = train_y[batch_no * batch_size : (batch_no + 1) * batch_size]
    train_on_batch(batch_train_x, batch_train_y)

print("\n\n\n\n")
print(ans)
for k in all_test_f1_weighted:
  print(k)

print("\n\n")
print(best_op)
