# -*- coding: utf-8 -*-

"""
Copyright 2018 NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import argparse
import os

import numpy as np
import tensorflow as tf

import nsml
from nsml import DATASET_PATH, HAS_DATASET, IS_ON_NSML

import model_helper as mh
from dataset import KinQueryDataset, preprocess


# DONOTCHANGE: They are reserved for nsml
# This is for nsml leaderboard
def bind_model(sess, config):
    # 학습한 모델을 저장하는 함수입니다.
    def save(dir_name, *args):
        # directory
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    # 저장한 모델을 불러올 수 있는 함수입니다.
    def load(dir_name, *args):
        saver = tf.train.Saver()
        # find checkpoint
        ckpt = tf.train.get_checkpoint_state(dir_name)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(dir_name, checkpoint))
        else:
            raise NotImplemented('No checkpoint!')
        print('Model loaded')

    def infer(raw_data, **kwargs):
        """

        :param raw_data: raw input (여기서는 문자열)을 입력받습니다
        :param kwargs:
        :return:
        """
        # dataset.py에서 작성한 preprocess 함수를 호출하여, 문자열을 벡터로 변환합니다
        preprocessed_data = preprocess(raw_data, config.strmaxlen)
        # 저장한 모델에 입력값을 넣고 prediction 결과를 리턴받습니다
        pred = sess.run(output_sigmoid, feed_dict={x: preprocessed_data, dropout_rate: 1.0})
        clipped = np.array(pred > config.threshold, dtype=np.int)
        # DONOTCHANGE: They are reserved for nsml
        # 리턴 결과는 [(확률, 0 or 1)] 의 형태로 보내야만 리더보드에 올릴 수 있습니다. 리더보드 결과에 확률의 값은 영향을 미치지 않습니다
        retval = list(zip(pred.flatten(), clipped.flatten()))
        print("Test result length: " + str(len(retval)))
        return retval

    # DONOTCHANGE: They are reserved for nsml
    # nsml에서 지정한 함수에 접근할 수 있도록 하는 함수입니다.
    nsml.bind(save=save, load=load, infer=infer)


def _batch_loader(iterable, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable: 데이터 list, 혹은 다른 포맷
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable)
    for n_idx in range(0, length, n):
        yield iterable[n_idx:min(n_idx + n, length)]


def _batch_loader(iterable_data, iterable_label, n=1):
    """
    데이터를 배치 사이즈만큼 잘라서 보내주는 함수입니다. PyTorch의 DataLoader와 같은 역할을 합니다

    :param iterable_data: 데이터 list, 혹은 다른 포맷
    :param iterable_label: label lest
    :param n: 배치 사이즈
    :return:
    """
    length = len(iterable_data)
    for n_idx in range(0, length, n):
        yield iterable_data[n_idx:min(n_idx + n, length)], iterable_label[n_idx:min(n_idx + n, length)]


def calculate_accuracy(infer_result, label):
    pred = np.array(infer_result > config.threshold, dtype=np.int)
    denominator = len(label)
    numerator = np.array(pred == label).sum()
    return numerator/float(denominator)


def partition_data_set(data_set, train_ratio: float):
    train_size = int(len(data_set) * train_ratio)
    shuffle_indices = np.random.permutation(np.arange(len(data_set)))
    shuffle_data, shuffle_label = data_set[shuffle_indices]
    return shuffle_data[:train_size], shuffle_label[:train_size], shuffle_data[train_size:], shuffle_label[train_size:]


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')

    # User options
    args.add_argument('--output', type=int, default=1)
    args.add_argument('--epochs', type=int, default=10)
    args.add_argument('--batch', type=int, default=2000)
    args.add_argument('--strmaxlen', type=int, default=400)
    args.add_argument('--filter_size', type=int, default=128)
    args.add_argument('--embedding', type=int, default=32)
    args.add_argument('--threshold', type=float, default=0.5)
    args.add_argument('--learning_rate', type=float, default=0.001)
    args.add_argument('--dropout', type=float, default=0.7)
    config = args.parse_args()

    print("default options:\nbatch: %d\nstrmaxlen: %d\nembedding: %d\nthreshold: %f\n" %
          (config.batch, config.strmaxlen, config.embedding, config.threshold))

    if not HAS_DATASET and not IS_ON_NSML:  # It is not running on nsml
        DATASET_PATH = '../sample_data/kin/'

    # 모델의 specification
    input_size = config.embedding*config.strmaxlen
    output_size = 1
    filter_size = config.filter_size
    learning_rate = config.learning_rate
    character_size = 251
    epsilon = 10e-7

    # n * strmaxlen(400)
    dropout_rate = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.int32, [None, config.strmaxlen])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    # 임베딩 251 * embedding(8)
    char_embedding = tf.get_variable('char_embedding', [character_size, config.embedding, 1])
    embedded = tf.nn.embedding_lookup(char_embedding, x)

    # conv2_kernel = mh.weight_variable("conv2", [2, word_vec_len, 1, filter_size])
    conv3_kernel = mh.weight_variable("conv3", [3, config.embedding, 1, filter_size])
    conv4_kernel = mh.weight_variable("conv4", [4, config.embedding, 1, filter_size])
    conv5_kernel = mh.weight_variable("conv5", [5, config.embedding, 1, filter_size])

    # conv2 = tf.nn.conv2d(x, conv2_kernel, strides=[1, 1, 1, 1], padding="VALID")
    # conv3 = tf.nn.conv2d(embedded, conv3_kernel, strides=[1, 1, 1, 1], padding="VALID")
    conv3 = tf.nn.conv2d(embedded, conv3_kernel, strides=[1, 1, 1, 1], padding="VALID")
    conv4 = tf.nn.conv2d(embedded, conv4_kernel, strides=[1, 1, 1, 1], padding="VALID")
    conv5 = tf.nn.conv2d(embedded, conv5_kernel, strides=[1, 1, 1, 1], padding="VALID")

    # bias2 = tf.Variable(tf.constant(0.1, shape=[filter_size]))
    bias3 = tf.Variable(tf.constant(0.1, shape=[filter_size]))
    bias4 = tf.Variable(tf.constant(0.1, shape=[filter_size]))
    bias5 = tf.Variable(tf.constant(0.1, shape=[filter_size]))

    # h2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
    h3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3))
    h4 = tf.nn.relu(tf.nn.bias_add(conv4, bias4))
    h5 = tf.nn.relu(tf.nn.bias_add(conv5, bias5))

    # p2 = tf.nn.max_pool(h2, ksize=[1, config.embedding - 2 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    p3 = tf.nn.max_pool(h3, ksize=[1, config.strmaxlen - 3 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    p4 = tf.nn.max_pool(h4, ksize=[1, config.strmaxlen - 4 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
    p5 = tf.nn.max_pool(h5, ksize=[1, config.strmaxlen - 5 + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

    # shape (1, 1, 1, 128)
    pooled_output = [p3, p4, p5]  # [p2, p3, p4, p5]

    full_conn_size = len(pooled_output) * filter_size
    h_pool = tf.concat(pooled_output, 3)

    # shape (1, 128 * len(pooled_output))
    h_pool_flat = tf.reshape(h_pool, [-1, full_conn_size])

    # dropout
    h_drop = tf.nn.dropout(h_pool_flat, dropout_rate)

    full_conn_w = mh.weight_variable("W", [full_conn_size, output_size])
    full_conn_bias = mh.bias_variable([output_size])
    output = tf.nn.xw_plus_b(h_drop, full_conn_w, full_conn_bias)
    output_sigmoid = tf.nn.sigmoid(output)

    output_sigmoid = tf.nn.sigmoid(output)

    # loss와 optimizer
    binary_cross_entropy = tf.reduce_mean(-(y_ * tf.log(output_sigmoid+epsilon)) - (1-y_) * tf.log(1-output_sigmoid+epsilon))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(binary_cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # DONOTCHANGE: Reserved for nsml
    bind_model(sess=sess, config=config)

    # DONOTCHANGE: Reserved for nsml
    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        # 데이터를 로드합니다.
        dataset = KinQueryDataset(DATASET_PATH, config.strmaxlen)
        dataset_len = len(dataset)
        # (phase1) Dataset size on server: 76520 (doubled, 38260 is actual size)
        # (phase2) Dataset size on server: 76526
        train_data, train_label, test_data, test_label = partition_data_set(dataset, 0.8)
        train_len = len(train_data)
        test_len = len(test_data)
        print('Train data size: ', train_len, ' Test data size: ', test_len)
        one_batch_size = dataset_len//config.batch
        if dataset_len % config.batch != 0:
            one_batch_size += 1
        # epoch마다 학습을 수행합니다.
        for epoch in range(config.epochs):
            avg_loss = 0.0
            for i, (data, labels) in enumerate(_batch_loader(train_data, train_label, config.batch)):
                _, loss = sess.run([train_step, binary_cross_entropy],
                                   feed_dict={x: data, y_: labels, dropout_rate: config.dropout})
                print('Batch : ', i + 1, '/', one_batch_size,
                      ', BCE in this minibatch: ', float(loss))
                avg_loss += float(loss)
            print('epoch:', epoch,
                  ' train_loss:', float(avg_loss/one_batch_size),
                  ' accuracy:',
                  calculate_accuracy(sess.run(output_sigmoid,
                                              feed_dict={x: test_data, y_: test_label, dropout_rate: 1.0}),
                                     test_label)
                  )
            nsml.report(summary=True, scope=locals(), epoch=epoch, epoch_total=config.epochs,
                        train__loss=float(avg_loss/one_batch_size), step=epoch)
            # DONOTCHANGE (You can decide how often you want to save the model)
            nsml.save(epoch)

    # 로컬 테스트 모드일때 사용합니다
    # 결과가 아래와 같이 나온다면, nsml submit을 통해서 제출할 수 있습니다.
    # [(0.3, 0), (0.7, 1), ... ]
    elif config.mode == 'test_local':
        with open(os.path.join(DATASET_PATH, 'train/train_data'), 'rt', encoding='utf-8') as f:
            queries = f.readlines()
        res = []
        for batch in _batch_loader(queries, config.batch):
            temp_res = nsml.infer(batch)
            res += temp_res

        with open(os.path.join(DATASET_PATH, 'train/train_label')) as f:
            labels = np.array([[np.float32(x)] for x in f.readlines()])
            predictions = np.array(res)[:,1]
            accuracy = np.array(predictions > config.threshold).sum()/float(len(labels))
            print('Accuracy: ', accuracy)