import tensorflow as tf
import pandas as pd
import numpy as np

train = pd.read_csv("optdigits.tra")
test = pd.read_csv("optdigits.tes")

def apply(y):
    ym = np.zeros([y.shape[0], 10])
    i = 0
    for each in y:
        ym[i, each] = 1
        i += 1
    return ym

Xtrain = train.iloc[:, :64]
ytrain = apply(train.iloc[:, 64])
Xtest = test.iloc[:, :64]
ytest = apply(test.iloc[:, 64])

x = tf.placeholder("float", [None, 64])
w = tf.Variable(tf.zeros([64, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
sess.run(train_step, feed_dict={x: Xtrain, y_: ytrain})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict = {x: Xtest, y_: ytest}))
