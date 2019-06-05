from __future__ import division
from __future__ import print_function
import time
import tensorflow as tf
from sklearn import metrics
from utils import *
from models import GCN, MLP
import random
import os
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'twitter', "twitter_hate_off"]
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")


# Set random seed
seed = random.randint(1, 200)
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 400, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.8, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(
    FLAGS.dataset)
# print(adj)

print("Features shape :: " , features.shape)
print("Features shape_0 :: " , features.shape[0])
# print("Features shape :: " , features.shape)
features = sp.identity(features.shape[0])  # featureless

# glove features

print(test_mask)


# exit(0)
print('embeddings:')
def loadGloveModel(gloveFile,words):
    print("Loading Glove Model")
    # path = os.path.join("data", "gcn_glove_" + dataset + ".json")
    # if os.path.exists(path):
    #     with open(path, "r") as f:
    #         return json.load(f)

    f = open(gloveFile,'r')

    all_words = set()
    for line in words:
        for word in line.split():
            all_words.add(word)
    
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0].strip().strip('<').strip('>')
        if word in all_words:
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print("Done.",len(model)," words loaded! Actual : ", len(all_words))
    
    # with open(path, "w") as f:
    #     json.dump(model, f)
    return model

dim = 100
f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()
glove_file_name = "./data/glove.twitter.27B."+ str(dim) +"d.txt"
glove_vectors = loadGloveModel(glove_file_name, words)
glove_embeddings = np.random.normal(size = (adj.shape[0], dim))
glove_embeddings[:train_size] = np.zeros(shape = (train_size, dim))
glove_embeddings[adj.shape[0] - test_size:] = np.zeros(shape = (test_size, dim))

for i in range(len(words)):
    word = words[i].strip()
    if word in glove_vectors:
        glove_embeddings[i + train_size, :] = glove_vectors[word]
    else:
        glove_embeddings[i + train_size] = np.random.normal(size = [dim])


word_embs = np.array(glove_embeddings)
print(adj.shape)
print(features.shape)
print(word_embs.shape)

# import cPickle as cp 
# with open('feature.data') as f:
#     cp.dump(features, f)

# Some preprocessing
features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    # helper variable for sparse dropout
    'num_features_nonzero': tf.placeholder(tf.int32)
}

# Create model
print("\n\n\n\n\nCreate model\n\n\n")
print(features[2][1])
model = model_func(
    placeholders, input_dim=features[2][1], logging=True, word_emb=word_embs.astype(np.float32), train_size = train_size, test_size = test_size)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)
# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(
        features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels, model.hidden1], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], outs_val[4],(time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())
cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(
        features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy,
                     model.hidden1], feed_dict=feed_dict)
    # Validation
    cost, acc, pred, labels, repre,duration = evaluate(
        features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")


# Training
train_cost, train_acc, pred, labels,t_representation, train_duration = evaluate(
    features, support, y_train, train_mask, placeholders)
print("\n\n\n\nTrain set results:", "cost=", "{:.5f}".format(train_cost),
      "accuracy=", "{:.5f}".format(train_acc), "time=", "{:.5f}".format(train_duration))

train_pred = []
train_labels = []
print(len(train_mask))
for i in range(len(train_mask)):
    if train_mask[i]:
        train_pred.append(pred[i])
        train_labels.append(labels[i])

print("Train Precision, Recall and F1-Score...")
print(metrics.classification_report(train_labels, train_pred, digits=4))
print("Macro average Train Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(train_labels, train_pred, average='macro'))
print("Micro average Train Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(train_labels, train_pred, average='micro'))
print("Weight average Train Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(train_labels, train_pred, average='weighted'))



# Testing
test_cost, test_acc, pred, labels,representation, test_duration = evaluate(
    features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

print(representation)
fp = open("out_pred.txt", "w")
for i,j in zip(test_pred, test_labels):
    fp.write(str(i) + "," + str(j) + "\n")
fp.close()

print(metrics.confusion_matrix(test_labels,test_pred))



print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))
print("Weight average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='weighted'))


# doc and word embeddings
print('embeddings:')
word_embeddings = outs[3][train_size: adj.shape[0] - test_size]
train_doc_embeddings = outs[3][:train_size]  # include val docs
test_doc_embeddings = outs[3][adj.shape[0] - test_size:]

print(len(word_embeddings), len(train_doc_embeddings),
      len(test_doc_embeddings))
print(word_embeddings)

f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data/' + dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

doc_vectors = []
doc_id = 0
doc_representation = []
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_representation.append('doc_' + str(i) + '\t' + str(t_representation[i]))
    doc_id += 1


f = open('data/' + dataset + '_doc_train_representation.txt', 'w')
f.write(str(doc_representation))
f.close()

doc_representation = []
for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_representation.append('doc_' + str(i) + '\t' + str(train_mask[i]) +'\t'+ str(representation[i]))
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)

f = open('data/' + dataset + '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()


f = open('data/' + dataset + '_doc_test_representation.txt', 'w')
f.write(str(doc_representation))
f.close()

import pickle
with open("saved_everything", "wb") as f:
    pickle.dump((train_doc_embeddings, test_doc_embeddings, t_representation, representation), f)
