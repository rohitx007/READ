import numpy as np
import random
import linecache
from tensorflow.keras.utils import Sequence
import pickle
import os
# from irl_generation.py import *
# SEQ_LENGTH = 12 # sequence length

save_dir = "save"
os.makedirs(save_dir, exist_ok=True)
save_dir = os.path.abspath(save_dir)

class Vocab:
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(file_path):
    '''
    # Arguments:
        file_path: str
    # Returns:
        data: list of list of str, data[i] means a sentence, data[i][j] means a
            word.
    '''
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()
        data.append(words)

    return data

# def sentence_to_ids(vocab, sentence, UNK=3):
#     '''
#     # Arguments:
#         vocab: SeqGAN.utils.Vocab
#         sentence: list of str
#     # Returns:
#         ids: list of int
#     '''
#     ids = [vocab.word2id.get(word, UNK) for word in sentence]
#     return ids

def pad_seq(seq, max_length, PAD=1):
    """
    :param seq: list of int,
    :param max_length: int,
    :return seq: list of int,
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq




class GeneratorPretrainingGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path: str, path to data x
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        x, y_true = generator.__getitem__(idx=11)
        print(x[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(y_true[0][0])
        >>> 0, 1, 0, 0, 0, 0, 0, ..., 0

        id2word = generator.id2word

        x_words = [id2word[id] for id in x[0]]
        print(x_words)
        >>> <S> I have a <UNK> </S> <PAD> ... <PAD>
    '''
    def __init__(self, path_union, path_seed, positive_file_union, positive_file_seed, T=12, min_count=1, shuffle=True):
        self.PAD = 1
        self.BOS = 0
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path_union = path_union
        self.path_seed = path_seed
        self.T = T
        self.min_count = min_count
        self.positive_file_union = positive_file_union
        self.positive_file_seed = positive_file_seed

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        sentences = load_data(path_union)
        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        
        # print(self.word2id)
        # print("_"*40)
        # print(self.id2word)
        # print("_"*40)
        # print(self.raw_vocab)
        # print("_"*40)
        # print(self.V)
        # print("_"*40)

        x = []
        max_length = T
        # print(sentences[:5])
        for sentence in sentences:
            # for words in sentence:
            ids = self.vocab.sentence_to_ids(sentence)
            # print("ids =  ", ids)
            # break

            ids_x = []

            ids_x.append(self.BOS)
            ids_x.extend(ids)
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            max_length = max(max_length, len(ids_x))

            if self.T is not None:
                max_length = self.T

            for i, ids in enumerate(x):
                x[i] = x[i][:max_length]
            
            x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)
        print(np.shape(x))

        # print(x)

        np.savetxt(positive_file_union, x, delimiter=" ", fmt="%d")

        saving_word2id_dic_path = os.path.join(save_dir,'word2id.pkl')
        with open(saving_word2id_dic_path, 'wb') as handle:
            pickle.dump(self.word2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        saving_id2word_path = os.path.join(save_dir,'id2word.pkl')
        with open(saving_id2word_path, 'wb') as handle:
            pickle.dump(self.id2word, handle, protocol=pickle.HIGHEST_PROTOCOL)
        saving_params_path = os.path.join(save_dir,'parameters.txt')
        with open(saving_params_path,"w") as params:
            params.write(str(self.V))

################Convert Seed data to Numbers##################

        sentences = load_data(path_seed)
        x = []
        max_length = T
        # print(sentences[:5])
        for sentence in sentences:
            # for words in sentence:
            ids = self.vocab.sentence_to_ids(sentence)
            # print("ids =  ", ids)
            # break

            ids_x = []

            ids_x.append(self.BOS)
            ids_x.extend(ids)
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            max_length = max(max_length, len(ids_x))

            if self.T is not None:
                max_length = self.T

            for i, ids in enumerate(x):
                x[i] = x[i][:max_length]
            
            x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)
        print(np.shape(x))

        # print(x)

        np.savetxt(positive_file_seed,x,delimiter=" ", fmt="%d")

if __name__ == '__main__':
    GeneratorPretrainingGenerator(path_union=os.path.join(save_dir,'union_seed_one_walk.txt'), path_seed = os.path.join(save_dir,'seed.txt'), positive_file_union = os.path.join(save_dir,'union_data.txt'), positive_file_seed = os.path.join(save_dir,'seed_data.txt'),  T=SEQ_LENGTH, min_count=1)