# -*- coding: utf-8 -*-

# 1つのパーセプトロン
# 中間層のない学習を行う
# 784のデータを10の出力にする


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


# 入力層を作っている
x = tf.placeholder(tf.float32, [None, 784])

# 重み
# 重みは前の層のユニットと次の層の積の数だけ作られる
W = tf.Variable(tf.zeros([784, 10]))

# バイアス
b = tf.Variable(tf.zeros([10]))

# 順伝搬
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 損失関数の定義(交差エントロピー関数を使う)
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 学習則
# 勾配降下法を用いる
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):   # 1000回学習
    batch_xs, batch_ys = mnist.train.next_batch(100) # ミニバッチサイズは100
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
