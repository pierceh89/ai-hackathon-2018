import os
import numpy as np
import re
from konlpy.tag import Twitter
import gensim
import multiprocessing
from pathlib import Path
special_chars = r'[`~@#$%^&*\(\)-_=\+\|\[\]{};:\'\",.<>/]'


def get_train_queries(queries_path):
    ret = []
    with open(queries_path, 'rt', encoding='utf8') as f:
        for pair_sentence in f.readlines():
            sentences = pair_sentence.split('\t')
            ret.append(normalize_sentences(sentences))
    return ret


def normalize_sentence(sentence):
    return re.sub(special_chars, "", sentence)


def normalize_sentences(sentences):
    ret = []
    for sentence in sentences:
        ret.append(normalize_sentence(sentence))
    return ret


config = {
            'min_count': 3,  # 등장 횟수가 5 이하인 단어는 무시
            'size': 100,  # 100차원짜리 벡터스페이스에 embedding
            'sg': 1,  # 0이면 CBOW, 1이면 skip-gram을 사용한다
            'batch_words': 400,  # 사전을 구축할때 한번에 읽을 단어 수
            'iter': 200,  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수
            'workers': multiprocessing.cpu_count()
        }


class KinQueryW2VDataSet:

    def __init__(self, model_name: str, dataset_path: str, max_length: int):
        """

        :param dataset_path: training set path
        :param max_length: max noun/token length
        """
        # 데이터, 레이블 각각의 경로
        queries_path = os.path.join(dataset_path, 'train', 'train_data')
        labels_path = os.path.join(dataset_path, 'train', 'train_label')
        train_set = get_train_queries(queries_path)

        # construct word embeddings
        self.w2v_loader = W2VLoader(model_name, train_set)
        self.queries = np.reshape(self.w2v_loader.embed_sentences(train_set, max_length), [-1, max_length, config['size'], 1])

        # 지식인 레이블을 읽고 preprocess까지 진행합니다.
        with open(labels_path) as f:
            self.labels = np.array([[np.float32(x)] for x in f.readlines()])

    def __len__(self):
        """

        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.queries)

    def __getitem__(self, idx):
        """

        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.queries[idx], self.labels[idx]


class W2VLoader:

    def __init__(self, model_name: str, dataset: list=None):
        """
        저장된 데이터셋이 있으면 로드하고, 없으면 트레이닝한다
        :param model_name: word2vec model name
        :param dataset_path: training data set path
        """
        # initialize twitter tokenizer
        self.twitter = Twitter()

        if model_name is '':
            return

        print("load w2v model... model_name: %s" % model_name)
        model_path = Path(model_name)
        if model_path.exists():
            # load word2vec model
            self.model = gensim.models.Word2Vec.load(model_name)
            self.word_vectors = self.model.wv
            self.vocab = self.word_vectors.vocab.keys()
            return

        if dataset is None:
            raise Exception('Cannot load data and load train data')

        model = self.learn_w2v(model_name, dataset)
        self.model = model
        self.word_vectors = self.model.wv
        self.vocab = self.word_vectors.vocab.keys()

    def learn_w2v(self, model_name, train_set):
        """

        :param model_name:
        :param train_set:
        :return:
        """
        # train_set = self.get_train_queries(queries_path)
        tokenized_sentences = []
        for pair_sentence in train_set:
            tokenized_sentences.append(self.twitter.morphs(pair_sentence[0]))
            tokenized_sentences.append(self.twitter.morphs(pair_sentence[1]))

        model = gensim.models.Word2Vec(**config)
        model.build_vocab(tokenized_sentences)
        model.train(tokenized_sentences,
                    total_examples=len(tokenized_sentences),
                    epochs=config['iter'])

        return model

    def embed_sentence(self, sentence, max_length):
        tokens = self.twitter.morphs(sentence)
        word_vecs = []
        for token in tokens:
            if token in self.vocab:
                word_vecs.append(self.word_vectors[token])
            else:
                word_vecs.append(np.zeros(config['size']))

        token_len = len(tokens)
        if token_len > max_length:
            return np.array(word_vecs[:max_length], dtype=np.float32)
        else:
            mat = np.zeros([max_length, config['size']])
            mat[:token_len] = np.array(word_vecs, dtype=np.float32)
            return mat

    def embed_sentences(self, sentences, max_length):
        ret = []
        for pair_sentence in sentences:
            ret.append(self.embed_sentence(' '.join(pair_sentence), max_length))
        return np.array(ret)

    def save_model(self, path):
        print('saving model in path: ' + path)
        self.model.save(path)

    def reload_model_from_path(self, path):
        print('loading model in path: ' + path)
        self.model = gensim.models.Word2Vec.load(path)
        self.word_vectors = self.model.wv
        self.vocab = self.word_vectors.vocab.keys()

    def set_model(self, model):
        self.model = model
        self.word_vectors = model.wv
        self.vocab = model.wv.vocab.keys()
