# -*- coding: utf-8 -*-

# 多層パーセプトロン
# 中間層が1つあって、中間層は100個のデータにする
# 784 -> 100 -> 10

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


# 入力層を作っている
x = tf.placeholder(tf.float32, [None, 784])

# 重み
# 重みは前の層のユニットと次の層の積の数だけ作られる
# ※中間層があるネットワークでは重みをzeroで初期化すると学習が進まない
# W1 = tf.Variable(tf.zeros([784, 100]))
W1 = tf.Variable(tf.truncated_normal([784,100], 0, 0.1))
b1 = tf.Variable(tf.zeros([100]))
y1 = tf.nn.relu(tf.matmul(x,W1) + b1)

# W2 = tf.Variable(tf.zeros([100, 10]))
W2 = tf.Variable(tf.truncated_normal([100,10], 0, 0.1))
b2 = tf.Variable(tf.zeros([10]))
y2 = tf.nn.softmax(tf.matmul(y1,W2) + b2)


# 損失関数の定義(交差エントロピー関数を使う)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y2), reduction_indices=[1]))

# 学習則
# 勾配降下法を用いる
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

correct_prediction = tf.equal(tf.argmax(y2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 学習させながらprintして段階的に学習状況を確認する
for i in range(1000):   # 1000回学習
    batch_xs, batch_ys = mnist.train.next_batch(100) # ミニバッチサイズは100
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

