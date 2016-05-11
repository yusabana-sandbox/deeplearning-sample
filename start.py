import tensorflow as tf

mat_x = tf.placeholder("float", shape=(2,2))
mat_y = tf.placeholder("float", shape=(2,2))


op_add = mat_x + mat_y
op_matmul = tf.matmul(mat_x, mat_y)

sess = tf.Session()
x_1 = [[1,2], [3,4]]
y_1 = [[5,6], [7,8]]

print sess.run(op_add, feed_dict={mat_x: x_1, mat_y: y_1})
print sess.run(op_matmul, feed_dict={mat_x: x_1, mat_y: y_1})
