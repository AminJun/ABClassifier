# Based on code from https://github.com/tensorflow/cleverhans
#
# This is the code for the paper
#
# Certifying Some Distributional Robustness with Principled Adversarial Training
# Link: https://openreview.net/forum?id=Hk6kPgZA-
#
# Authors: Aman Sinha, Hongseok Namkoong, John Duchi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import keras
import numpy as np
import tensorflow as tf
from keras.backend import manual_variable_initialization
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

from attacks import ABAttacker, PGDAttacker
from utils import cnn_model
from utils_mnist import data_mnist
from utils_tf import model_train, model_eval

FLAGS = flags.FLAGS

flags.DEFINE_integer('nb_epochs', 25, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 256, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
flags.DEFINE_string('train_dir', '.', 'Training directory')
flags.DEFINE_string('filename_erm', 'erm.h5', 'Training directory')
flags.DEFINE_string('filename_wrm', 'wrm.h5', 'Training directory')

train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
}
eval_params = {'batch_size': FLAGS.batch_size}

seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

_LABELS = [3, 8]
_NB_LOGITS = 2
_BETA = 1


def filter_by_label(x, y, labels=_LABELS):
    index_filter = np.isin(np.argmax(y, axis=1), labels)
    y = y[index_filter]
    y = y[:, labels]
    return x[index_filter], y


def get_barrier(distance_boundary):
    return tf.exp(-_BETA * distance_boundary)


def main(argv=None):
    keras.layers.core.K.set_learning_phase(1)
    manual_variable_initialization(True)
    tf.reset_default_graph()

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    # Get MNIST test data
    x_train, y_train, x_test, y_test = data_mnist()
    x_train, y_train = filter_by_label(x_train, y_train)
    x_test, y_test = filter_by_label(x_test, y_test)

    assert y_train.shape[1] == _NB_LOGITS
    label_smooth = .1
    y_train = y_train.clip(label_smooth, 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, _NB_LOGITS))

    # Define TF model graph
    model = cnn_model(activation='elu', nb_classes=_NB_LOGITS)
    predictions = model(x)

    pgd = PGDAttacker(model, back='tf', sess=sess, eps=0.3, eps_iter=0.05, n_iter=5)
    adv_x = pgd.generate(x, y)
    predictions_adv_x = model(adv_x)

    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        accuracy = model_eval(sess, x, y, predictions, x_test, y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

        # Accuracy of the model on PGD adversarial examples
        accuracy_pgd = model_eval(sess, x, y, predictions_adv_x, x_test, y_test, args=eval_params)
        print('Test accuracy on PGD examples: %0.4f\n' % accuracy_pgd)

    # Train the model
    model_train(sess, x, y, predictions, x_train, y_train, evaluate=evaluate, args=train_params,
                save=False)
    model.model.save(FLAGS.train_dir + '/' + FLAGS.filename_erm)

    print('')
    print("Repeating the process, using ABClassifier adversarial training")
    # Redefine TF model graph
    model_adv = cnn_model(activation='elu')
    predictions_clean = model_adv(x)
    distance_boundary = ABAttacker(model_adv, sess=sess, L=10, C=10 * 28).get_distance(x, steps=15)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y) + get_barrier(
        distance_boundary)

    pgd = PGDAttacker(model_adv, back='tf', sess=sess, eps=0.3, eps_iter=0.05, n_iter=5)
    adv_x = pgd.generate(x, y)
    predictions_adv_x = model_adv(adv_x)

    # wrm2 = WassersteinRobustMethod(model_adv, sess=sess)
    # predictions_adv_adv_wrm = model_adv(wrm2.generate(x, **distance_params_params))

    def evaluate_adv():
        # Accuracy of adversarially trained model on legitimate test inputs
        accuracy = model_eval(sess, x, y, predictions_clean, x_test, y_test, args=eval_params)
        print('Test accuracy on legitimate test examples: %0.4f' % accuracy)

        # Accuracy of the adversarially trained model on Wasserstein adversarial examples
        accuracy_adv_wass = model_eval(sess, x, y, predictions_adv_x, x_test, y_test,
                                       args=eval_params)
        print('Test accuracy on PGD examples: %0.4f\n' % accuracy_adv_wass)

    model_train(sess, x, y, predictions_clean, x_train, y_train,
                predictions_adv=predictions_adv_x, evaluate=evaluate_adv, args=train_params,
                save=False, loss=loss)
    model_adv.model.save(FLAGS.train_dir + '/' + FLAGS.filename_wrm)


if __name__ == '__main__':
    app.run()
