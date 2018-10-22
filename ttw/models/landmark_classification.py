# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score

class LandmarkClassifier(nn.Module):

    def __init__(self, textrecog_features, fasttext_features, resnet_features, num_tokens=100, pool='sum', resnet_dim=2048, fasttext_dim=300):
        super().__init__()
        self.textrecog_features = textrecog_features
        self.fasttext_features = fasttext_features
        self.resnet_features = resnet_features
        self.pool = pool
        self.num_classes = 10

        if self.fasttext_features:
            self.fasttext_linear = nn.Linear(fasttext_dim, self.num_classes, bias=False)
        if self.resnet_features:
            self.resnet_linear = nn.Linear(resnet_dim, self.num_classes, bias=False)
        if self.textrecog_features:
            self.embed = nn.Embedding(num_tokens, 50, padding_idx=0)
            self.textrecog_linear = nn.Linear(50, self.num_classes, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        batchsize = batch['target'].size(0)
        logits = Variable(torch.FloatTensor(batchsize, self.num_classes)).zero_()

        if self.textrecog_features:
            embeddings = self.embed(batch['textrecog'])
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.textrecog_linear(embeddings[i, :, :]).sum(dim=0)
                else:
                    logits[i, :] += self.textrecog_linear(embeddings[i, :, :]).max(dim=0)[0]

        if self.fasttext_features:
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.fasttext_linear(batch['fasttext'][i, :, :]).sum(dim=0)
                else:
                    logits[i, :] += self.fasttext_linear(batch['fasttext'][i, :, :]).max(dim=0)[0]

        if self.resnet_features:
            for i in range(batchsize):
                if self.pool == 'sum':
                    logits[i, :] += self.resnet_linear(batch['resnet'][i, :, :]/10).sum(dim=0)
                else:
                    logits[i, :] += self.resnet_linear(batch['resnet'][i, :, :]/10).max(dim=0)[0]

        self.loss.weight = batch['weight'].view(-1).data
        out = dict()
        batch['target'] = batch['target'].float()
        target = batch['target'].view(-1)
        out['loss'] = self.loss(logits.view(-1), target)

        y_pred = torch.ge(self.sigmoid(logits), 0.5).float().data.numpy()
        y_true = batch['target'].data.numpy()

        out['f1'] = f1_score(y_true, y_pred, average='weighted')
        out['precision'] = precision_score(y_true, y_pred, average='weighted')
        out['recall'] = recall_score(y_true, y_pred, average='weighted')

        return out
