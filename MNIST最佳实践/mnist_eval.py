import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference
import mnist_train

# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-input')
        global_step = tf.Variable(0, trainable=False)
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        y_output=tf.argmax(y, 1)
        y_right_ans=tf.argmax(y_, 1)
        
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt :
                saver = tf.train.import_meta_graph('H:/Tensorflow_learning/MNIST_Practice/MNIST_model/mnist_model-1.meta')
                saver.restore(sess, tf.train.latest_checkpoint('H:/Tensorflow_learning/MNIST_Practice/MNIST_model/'))
                accuracy_score, steps, y_out,y_right = sess.run([accuracy, global_step,y_output,y_right_ans], feed_dict=validate_feed)
                print("After %s training step(s), validation accuracy = %g" % (steps, accuracy_score))
                print(y_out,y_right)

            else:
                print('No checkpoint file found')
                return
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets("../../../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    main()