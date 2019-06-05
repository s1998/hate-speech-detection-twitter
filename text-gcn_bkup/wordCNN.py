
import tensorflow as tf
import numpy as np

class WordCNN:
    def __init__(self, 
            vocabulary_size, document_max_len, num_class, wordVecs = None, embedding_size = 100, doc_embs = False):
        self.embedding_size = embedding_size
        self.learning_rate = 1e-3
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 100

        self.x = tf.placeholder(tf.int32, [None, document_max_len], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.doc_embs = tf.placeholder(tf.float32, [None, 200])
        self.is_training = tf.placeholder(tf.bool, [], name="is_training")
        self.global_step = tf.Variable(0, trainable=False)
        self.keep_prob = tf.placeholder(dtype = tf.float32)

        with tf.name_scope("embedding"):
            if wordVecs is None:
                self.embeddings = tf.get_variable(
                    "embeddings", 
                    [vocabulary_size-1, self.embedding_size], 
                    initializer = tf.contrib.layers.xavier_initializer())
            else:
                self.embeddings = tf.get_variable("embeddings", 
                         initializer=wordVecs, dtype = tf.float32)

            self.embeddings = tf.concat([
                self.embeddings,
                tf.get_variable(
                    "ukn_embeddings", 
                    [1, self.embedding_size], 
                    initializer = tf.contrib.layers.xavier_initializer()),
                tf.constant(np.zeros((1, self.embedding_size), dtype=np.float32))], 0) 
            self.x_emb = tf.nn.embedding_lookup(self.embeddings, self.x)
            self.x_emb = tf.expand_dims(self.x_emb, -1)
        pooled_outputs = []
        for filter_size in self.filter_sizes:
            conv = tf.layers.conv2d(
                self.x_emb,
                filters=self.num_filters,
                kernel_size=[filter_size, self.embedding_size],
                strides=(1, 1),
                padding="VALID",
                activation=tf.nn.relu)
            pool = tf.layers.max_pooling2d(
                conv,
                pool_size=[document_max_len - filter_size + 1, 1],
                strides=(1, 1),
                padding="VALID")
            pooled_outputs.append(pool)

        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.contrib.layers.batch_norm(tf.reshape(h_pool, [-1, self.num_filters * len(self.filter_sizes)]))
        
        if doc_embs:
            self.doc_embs2 = tf.contrib.layers.batch_norm(tf.layers.dense(self.doc_embs, 10, activation=tf.nn.relu, use_bias = False))
            h_pool_flat = tf.concat([h_pool_flat, self.doc_embs2], axis = 1)

        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(self.h_drop, num_class, activation=lambda x : x)
            self.predictions = tf.argmax(self.logits, -1, output_type=tf.int32)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, self.y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.saver = tf.train.Saver()

    def train_on_batch(self, x, y, doc_embs):
        self.sess.run(self.optimizer, 
            feed_dict = {self.x : x, self.y : y, self.is_training : True, self.keep_prob : 0.5, self.doc_embs : doc_embs})

    def test_on_batch(self, x, doc_embs):
        return self.sess.run(self.predictions, 
            feed_dict = {self.x : x, self.is_training : False, self.keep_prob : 1.0, self.doc_embs : doc_embs})

    def return_predictions(self, x, doc_embs):
        return self.sess.run(self.predictions, 
            feed_dict = {self.x : x, self.is_training : False, self.keep_prob : 1.0, self.doc_embs : doc_embs})

    def return_h(self, x, doc_embs):
        return self.sess.run(self.h_drop, 
            feed_dict = {self.x : x, self.is_training : False, self.keep_prob : 1.0, self.doc_embs : doc_embs})

    def save_model(self, path):
        print("Saved : ", path)
        self.saver.save(self.sess, path, global_step = 0)

    def load_model(self, path):
        # saver = tf.train.import_meta_graph(path + "-0.meta")
        print("Loaded : ", path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint("./saved/"))
