import tensorflow as tf


def xavier_init(n_inputs, n_outputs, uniform=True):

    if uniform:
        # 6 was used in the paper
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        # 3 gives us approximately the same limits as above since this repicks
        # values greater than 2 standard deviations from the mean
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


def he_init(n_inputs):

    init_range = tf.sqrt(2.0 / n_inputs)
    return tf.random_uniform_initializer(-init_range, init_range)