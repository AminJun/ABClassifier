# Based on code from https://github.com/tensorflow/cleverhans
#
# This is the code for the paper
#
# Certifying Some Distributional Robustness with Principled Adversarial Training
# Link: https://openreview.net/forum?id=Hk6kPgZA-
#
# Authors: Aman Sinha, Hongseok Namkoong, John Duchi

from abc import ABCMeta

import numpy as np
import tensorflow as tf

from attacks_tf import wrm


class Attack:
    """
    Abstract base class for all attack classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, back='tf', sess=None):
        """
        :param model: A function that takes a symbolic input and returns the
                      symbolic output for the model's predictions.
        :param back: The backend to use. Either 'tf' (default) or 'th'.
        :param sess: The tf session to run graphs in (use None for Theano)
        """
        if not (back == 'tf' or back == 'th'):
            raise ValueError("Backend argument must either be 'tf' or 'th'.")
        if back == 'tf' and sess is None:
            raise Exception("A tf session was not provided in sess argument.")
        if back == 'th' and sess is not None:
            raise Exception("A session should not be provided when using th.")
        if not hasattr(model, '__call__'):
            raise ValueError("model argument must be a function that returns "
                             "the symbolic output when given an input tensor.")

        # Prepare attributes
        self.model = model
        self.back = back
        self.sess = sess
        self.inf_loop = False


def step_size_iter(max_step):
    for step in range(max_step):
        yield 1 / np.sqrt(2 + step)


class ABAttacker(Attack):
    def __init__(self, model, back='tf', sess=None, L=10, C=10 * 28):
        super(ABAttacker, self).__init__(model, back, sess)
        self.L = L
        self.C = C

    def get_distance(self, x, steps=15):
        predicts = self.model(x)

        # Compute loss
        cur_x = x
        cur_lambda = self.L / 2

        for step_size in step_size_iter(steps):
            grad_min = self.get_grad_min(cur_lambda, cur_x, predicts, x)
            cur_x = cur_x - grad_min * step_size
            grad_max = self.get_decision_boundary(cur_x)
            cur_lambda = tf.clip_by_value(cur_lambda + grad_max * step_size, -self.L, self.L)
        return (self.C * tf.square(tf.norm(cur_x - x)) +
                cur_lambda * self.get_decision_boundary(cur_x)) / self.C

    def get_grad_min(self, cur_lambda, cur_x, predicts, x):
        grad_db = cur_lambda * tf.gradients(self.get_decision_boundary(predicts), x)[0]
        grad_distance = 2 * self.C * (cur_x - x)
        return grad_db + grad_distance

    @staticmethod
    def get_decision_boundary(predicts):
        prob = tf.nn.softmax(predicts)
        return prob[0] - prob[1]


class PGDAttacker(Attack):
    def __init__(self, model, back='tf', sess=None, eps=0.3, eps_iter=0.05, n_iter=100):
        super(PGDAttacker, self).__init__(model=model, back=back, sess=sess)
        self.eps = eps
        self.eps_iter = eps_iter
        self.n_iter = n_iter

    def generate(self, x, y):
        adv_x = x
        for i in range(self.n_iter):
            print('=============' + str(i) + '============')
            loss_x = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.model(adv_x))
            grad_x = tf.gradients(loss_x, x)[0]
            adv_x = adv_x + self.eps_iter * tf.sign(grad_x)
            delta = tf.clip_by_value(adv_x - x, -self.eps, self.eps)
            adv_x = tf.clip_by_value(x + delta, 0, 1)
        return adv_x


class WassersteinRobustMethod(Attack):
    def __init__(self, model, back='tf', sess=None):
        super(WassersteinRobustMethod, self).__init__(model, back, sess)

    def generate(self, x, **kwargs):
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)
        return wrm(x, self.model(x), y=self.y, eps=self.eps, model=self.model, steps=self.steps)

    def parse_params(self, eps=0.3, ord=2, y=None, steps=15, **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:
        :param eps: (optional float) .5/gamma (Lagrange dual parameter)
        in the ICLR paper (see link above),
        :param ord: (optional) Order of the norm (mimics Numpy).
        Possible values: 2.
        :param y: (optional) A placeholder for the model labels. Only provide
        this parameter if you'd like to use true labels when crafting
        adversarial samples. Otherwise, model predictions are used as
        labels to avoid the "label leaking" effect (explained in this
        paper: https://arxiv.org/abs/1611.01236). Default is None.
        Labels should be one-hot-encoded.
        :param steps: how many gradient ascent steps to take in finding
        the adversarial example
        """
        self.eps = eps
        self.y = y
        self.steps = steps
        return True
