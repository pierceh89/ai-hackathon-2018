import tensorflow as tf

from initializer import xavier_init


def weight_variable(name, shape):
    initial = tf.get_variable(name=name, shape=shape, initializer=xavier_init(shape[0], shape[1]))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def hidden_layer(name, s1, s2, former_layer, dropout=None):
    next_layer = matmul(name, s1, s2, former_layer)
    if dropout is not None:
        next_layer = tf.nn.dropout(next_layer, dropout)
    return tf.nn.sigmoid(next_layer)


def out_layer(name, s1, s2, former_layer):
    out = matmul(name, s1, s2, former_layer)
    return out


def matmul(name, s1, s2, former_layer):
    weight = weight_variable(name, [s1, s2])
    bias = bias_variable([s2])
    next_layer = tf.matmul(former_layer, weight) + bias
    return next_layer

