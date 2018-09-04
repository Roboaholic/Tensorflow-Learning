import time
import tensorflow as tf
import MNIST_Inference
import MNIST_Training

from tensorflow.examples.tutorials.mnist import input_data

EVAL_INTERVAL_SECONDS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, MNIST_Inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, MNIST_Inference.OUTPUT_NODE], name='y-input')

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = MNIST_Inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(MNIST_Training.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(MNIST_Training.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path\
                        .split('/')[-1].split('-')[-1]
                    accuracy_score = sess.rin(accuracy, feed_dict=validate_feed)
                    print("after %s steps, accuracy is %g" % (global_step, accuracy_score))
                else:
                    print("no checkpoint file found")
                    return
            time.sleep(EVAL_INTERVAL_SECONDS)



def main(argv=None):
    mnist = input_data.read_data_sets("H:/path/to/MNIST_data/", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
