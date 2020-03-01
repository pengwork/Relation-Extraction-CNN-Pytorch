import pickle
import torch

from torch import nn
from pytorch_transformers import BertModel


class Net(torch.nn.Module):

    def __init__(self, args, params):
        super(Net, self).__init__()

        self.class_num = 19
        self.bert=args.bert

        if self.bert:
            word_emb_dim = 768
            self.word_emb = BertModel.from_pretrained('bert-base-uncased')
        else:
            with open(args.embedding_pkl_path + '_numpy.pkl', 'rb') as f:
                pretrained_weight = pickle.load(f)
                word_emb_dim = pretrained_weight.shape[1]
                self.word_emb = nn.Embedding(pretrained_weight.shape[0], pretrained_weight.shape[1])

        self.pos1_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        self.pos2_emb = nn.Embedding(params.pos_emb_size, params.pos_emb_dim, padding_idx=0)
        feature_dim = word_emb_dim + params.pos_emb_dim * 2

        self.convs = nn.ModuleList([nn.Conv1d(feature_dim, params.kernel_num, kernel_size=kernel_size) \
                                    for kernel_size in params.kernel_sizes])

        self.dropout = nn.Dropout(params.dropout_ratio)
        self.fc = nn.Linear(params.kernel_num * len(params.kernel_sizes), self.class_num)

        self.loss = nn.CrossEntropyLoss()

        if args.gpu:
            self.cuda()

    def forward(self, words, pos1, pos2):

        if self.bert:
            # with torch.no_grad():

            words = self.word_emb(words)[0]
        else:
            words = self.word_emb(words)
        words = words[:, 1:, :]  # 去除第一个 [cls]
        pos1 = self.pos1_emb(pos1)
        pos2 = self.pos2_emb(pos2)
        featrues = torch.cat([words, pos1, pos2], dim=2).permute(0, 2, 1)

        x = [torch.max(torch.relu(oneconv(featrues)), 2)[0] for oneconv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
