import json
from copy import deepcopy
from functools import partial

import pandas as pd
import pytorch_metric_learning.distances as pml_distances
import torch
from grad_cache import GradCache
from pytorch_metric_learning.losses import NTXentLoss
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

CORPUS_PATH = 'scifact/corpus.jsonl'
QUERIES_PATH = 'scifact/queries.jsonl'
QREL_TRAIN_PATH = 'scifact/qrels/train.tsv'
QREL_TEST_PATH = 'scifact/qrels/test.tsv'

MODEL_NAME = 'bert-base-uncased'


class SciFactDataset(Dataset):
    def __init__(self, tokenizer):
        corpus = dict()
        queries = dict()
        self._pairs = []
        with open(CORPUS_PATH) as f:
            for line in f:
                doc = json.loads(line)
                corpus[doc['_id']] = doc['title']
        with open(QUERIES_PATH) as f:
            for line in f:
                doc = json.loads(line)
                queries[doc['_id']] = doc['text']
        df = pd.read_csv(QREL_TRAIN_PATH, sep='\t')
        for i, (qid, cid, s) in df.iterrows():
            assert s == 1
            t1 = queries[str(qid)]
            t2 = corpus[str(cid)]
            self._pairs.append([t1, t2])

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        return self._pairs[idx]


class InfoNCELoss(nn.Module):
    def __init__(self):
        super(InfoNCELoss, self).__init__()
        self._loss = NTXentLoss(
            distance=pml_distances.CosineSimilarity(), temperature=0.07
        )

    def forward(self, embeddings_left, embeddings_right):
        labels = torch.arange(
            start=0, end=embeddings_left.shape[0], device=embeddings_left.device
        )
        return self._loss(
            labels=labels,
            ref_labels=deepcopy(labels),
            embeddings=embeddings_left,
            ref_emb=embeddings_right,
        )


class MyModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self._model = model

    def forward(self, *args, **kwargs):
        print('model input shape', kwargs['input_ids'].shape)
        return self._model(*args, **kwargs)


def train():
    model = MyModel(AutoModel.from_pretrained(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    loss_fn = InfoNCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    gc = GradCache(
        models=[model, model],
        chunk_sizes=5,
        loss_fn=loss_fn,
        get_rep_fn=lambda v: v.pooler_output,
    )

    dataset = SciFactDataset(tokenizer)
    collate_fn = lambda x: map(
        partial(tokenizer, return_tensors='pt', padding='max_length'), zip(*x)
    )
    dl = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn)

    for i, (input1, input2) in enumerate(dl):
        print('batch shape', input1['input_ids'].shape)
        optimizer.zero_grad()
        gc(input1, input2)
        optimizer.step()


if __name__ == '__main__':
    train()
