import torch
import pickle
import argparse

from pytorch_transformers import BertTokenizer
from torchtext.data import Field, TabularDataset
from torchtext.data import BucketIterator


class Dataset(object):
    def __init__(self, args, params):
        self.batch_size = params.batch_size
        self.fix_length = params.fix_length
        self.root_path = args.data_dir
        self.use_bert = args.bert

        if not self.use_bert:
            with open(args.embedding_pkl_path + '_word2idx.pkl', 'rb') as f:
                word2idx = pickle.load(f)
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        def word_tokenize(sentence):
            if self.use_bert:
                # tokenized_text = tokenizer.tokenize(sentence)    #会切分单词 导致不对应
                tokenized_text = sentence.split(' ')
                sentence = tokenizer.convert_tokens_to_ids(tokenized_text)
                # sentence = tokenizer.add_special_tokens_single_sentence(sentence)
                sentence = [101] + sentence
            else:

                tokenized_text = sentence.split(' ')
                sentence = [word2idx.get(word, 0) for word in tokenized_text]
                sentence = [0] + sentence  # 与bert统一

            return sentence

        def pos_tokenize(posids):
            return [int(_) for _ in posids.split(' ')]

        # dtype = torch.cuda.LongTensor if args.gpu and torch.cuda.is_available() else torch.int64

        TEXT = Field(sequential=True, tokenize=word_tokenize,
                     use_vocab=False, batch_first=True,
                     fix_length=self.fix_length + 1,  # 添加了 cls
                     pad_token=0)
        POSITION = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                         pad_token=0, batch_first=True, include_lengths=True)
        POSITION_NO_LEN = Field(sequential=True, tokenize=pos_tokenize, use_vocab=False, fix_length=self.fix_length,
                                pad_token=0, batch_first=True)
        LABEL = Field(sequential=False, use_vocab=False, batch_first=True)

        fields = {
            'sentence': ('words', TEXT),
            'label': ('label', LABEL),
            'e1': ('pos_e1', POSITION),
            'e2': ('pos_e2', POSITION_NO_LEN)
        }

        self.train, self.valid = TabularDataset.splits(
            path=self.root_path,
            train='train.txt', validation='test.txt',
            format='json',
            skip_header=False,
            fields=fields)

    def get_data(self, name='training'):
        if name == 'training':
            return BucketIterator(self.train, batch_size=self.batch_size, shuffle=True, repeat=False)
        elif name == 'validation':
            return BucketIterator(self.valid, batch_size=self.batch_size, shuffle=True, repeat=True)
        elif name == 'test':
            return BucketIterator(self.valid, batch_size=self.batch_size, shuffle=True, repeat=False)


class BatchWrapper(object):
    """对batch做个包装，方便调用，可选择性使用"""

    def __init__(self, dl, gpu):
        self.dl = dl
        self.gpu = gpu  # 是否使用gpu

    def __iter__(self):
        for batch in self.dl:
            words = getattr(batch, 'words')
            labels = getattr(batch, 'label')

            pos1s = getattr(batch, 'pos_e1')[0]
            lens = getattr(batch, 'pos_e1')[1]
            pos2s = getattr(batch, 'pos_e2')

            func = lambda x: x.cuda()
            if not self.gpu:
                yield [words, pos1s, lens, pos2s, labels]
            else:
                yield list(map(func, [words, pos1s, lens, pos2s, labels]))

    def __len__(self):
        return len(self.dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/', help="Directory containing the dataset")
    parser.add_argument('--embedding_pkl_path', default='./data/word_embedding', help="Path to word vecfile.")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--bert', default=False, help=" use Bert or wordembedding")

    params = type('classA', (object,), dict(batch_size=32, fix_length=96))()

    args = parser.parse_args()
    dataset = Dataset(args=args, params=params)

    val_data = BatchWrapper(dataset.get_data('validation'), gpu=True)
    batch = next(iter(val_data))
    print(batch)
    # print('batch:\n', batch)
    # print('batch_text:\n', batch.text)
    # print('batch_label:\n', batch.label)
