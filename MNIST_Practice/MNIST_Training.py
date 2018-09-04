import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import MNIST_Inference
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "H:/tensorflow_learning/MNIST_Practice/model/"
MODEL_NAME = "minist_model.ckpt"


def train(mnist):
    x = tf.placeholder(tf.float32, [None, MNIST_Inference.INPUT_NODE], name='x-input')  # 存BATCH_SIZE行，786列的数据，None即可变
    y_ = tf.placeholder(tf.float32, [None, MNIST_Inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = MNIST_Inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable=False)


# 下面创建滑动平均值

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # argmax函数可以提供答案编号，
    cross_entropy_mean = tf.reduce_mean(cross_entropy)                                                 # 即答案
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))     # 正则化
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    output = (tf.argmax(y, 1))
    right_ans = (tf.argmax(y_, 1))
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step, out, right_answer = sess.run(
                [train_op, loss, global_step, output, right_ans], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                print(out)
                print(right_answer)


def main(argv=None):
    mnist = input_data.read_data_sets("H:/path/to/datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
