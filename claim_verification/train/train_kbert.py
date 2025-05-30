"""
Train KBERT for RTE task
"""
# 读取文件
import sys
import json
import jsonlines

# 参数
import argparse

# NN
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
import collections
import random
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Pool

import brain
from brain.conceptnet_with_rank import ConceptNetWithRank
from brain import KnowledgeGraph
# uer utils
from uer.utils.tokenizer import *
from uer.model_builder import build_model
from uer.utils.optimizers import BertAdam
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model

from claim_verification.utils.Datasets import SciFactLabelPredictionDataset
from claim_verification.utils.Functions import printf


import requests
from collections import Counter
import pickle
import os
import string
import nltk
from nltk import pos_tag, WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
from collections import Counter
from tqdm import tqdm

class BertClassifier(nn.Module):
    def __init__(self, args, model):
        super(BertClassifier, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, args.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.NLLLoss()
        self.use_vm = False if args.no_vm else True
        print("[BertClassifier] use visible_matrix: {}".format(self.use_vm))

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size]
            mask: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        if not self.use_vm:
            vm = None
        output = self.encoder(emb, mask, vm)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        loss = self.criterion(self.softmax(logits.view(-1, self.labels_num)), label.view(-1))
        return loss, logits

def add_knowledge_worker(data, kg, vocab, args):
    # 原函数直接根据数据一行一行分析，但是也可以不这样做
    sentences_num = len(data['claim'])
    dataset = []

    for i in range(sentences_num):
        # {"claim":, "rationale":, "label":}
        try:

            claim, rationale, label = data['claim'][i], data['rationale'][i], int(data['label'][i])

            text = CLS_TOKEN + rationale + " " + SEP_TOKEN + " " + claim + SEP_TOKEN

            tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
            tokens = tokens[0]
            pos = pos[0]
            vm = vm[0].astype("bool")

            token_ids = [vocab.get(t) for t in tokens]
            mask = []
            seg_tag = 1
            for t in tokens:
                if t == PAD_TOKEN:
                    mask.append(0)
                else:
                    mask.append(seg_tag)
                if t == SEP_TOKEN:
                    seg_tag += 1

            dataset.append((token_ids, label, mask, pos, vm))

        except:
            print("Error line: ", str(i+1))

    return dataset


def trans_to_tensor(batch):
    input_ids = torch.LongTensor([example[0] for example in batch])
    label_ids = torch.LongTensor([example[1] for example in batch])
    mask_ids = torch.LongTensor([example[2] for example in batch])
    pos_ids = torch.LongTensor([example[3] for example in batch])
    vms = [example[4] for example in batch]
    return input_ids, label_ids, mask_ids, pos_ids, vms

def evaluate(args, model, dataset, kg, vocab):
    model.eval()
    targets = []
    outputs = []
    batch_size = args.batch_size
    instance_num = len(dataset)
    cnt = 0
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    print("Evaluating on {} instances".format(instance_num))
    with torch.no_grad():
        for i,batch in enumerate(DataLoader(dataset, batch_size=args.batch_size)):
            example = add_knowledge_worker(batch, kg, vocab, args)
            input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch = trans_to_tensor(example)
            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)

            try:
                loss, logits = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch)
            except:
                print(input_ids_batch)
                print(input_ids_batch.size())
                print(vms_batch)
                print(vms_batch.size())

            logits = nn.Softmax(dim=1)(logits)

            pred = torch.argmax(logits, dim=1)
            gold = label_ids_batch
            for j in range(pred.size()[0]):
                confusion[pred[j], gold[j]] += 1
            correct += torch.sum(pred==gold).item()
            cnt += pred.size()[0]
            targets.extend(gold.tolist())
            outputs.extend(pred.tolist())

    return {
        'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
        'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
        'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
        'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None)),
        'accuracy': correct/cnt,
        'confusion': confusion
    }

def build_word_ls(args):
    set_seed(args.seed)

    # --------------------load vocabulary--------------------
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # --------------------Build bert models--------------------
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)
    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained models.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    # Build classification models.
    model = BertClassifier(args, model)

    # --------------------Set the Device--------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.". format((torch.cuda.device_count())))
        model = nn.DataParallel(model)
    model = model.to(device)

    # --------------------Build knowledge Graph--------------------
    if args.kg_name == "ConceptNetWithRank":
        spo_files = args.kg_files
        kg = ConceptNetWithRank(spo_files, vocab_path=args.vocab_path, predicate=True)
    elif args.kg_name == "none":
        spo_files = []
        kg = KnowledgeGraph(spo_files=spo_files, vocab_path=args.vocab_path,predicate=True)
    else:
        spo_files = [args.kg_name]
        kg = KnowledgeGraph(spo_files=spo_files,vocab_path=args.vocab_path,predicate=True)

    # --------------------Read Data--------------------
    trainset = SciFactLabelPredictionDataset(args.corpus_path, args.train_path)
    devset = SciFactLabelPredictionDataset(args.corpus_path, args.dev_path)

    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None


    stop_words = set(stopwords.words('english'))
    wnl = WordNetLemmatizer()
    all_words = []
    for sample in trainset:
        print(sample.keys())
        line = sample['rationale'] + " " + sample['claim']
        words = nltk.word_tokenize(line.lower())  # 将文本转换为小写，并将其拆分为单词列表
        tagged_sent = pos_tag(words)
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
        words = [word for word in lemmas_sent if word.isalpha() and word not in stop_words and len(word)>1 ][1:] # 过滤掉非单词和停用词
        all_words.extend(words)

    for sample in devset:
        line = sample['rationale'] + " " + sample['claim']
        words = nltk.word_tokenize(line.lower())  # 将文本转换为小写，并将其拆分为单词列表
        words = [word for word in words if word.isalpha() and word not in stop_words and len(word)>1][1:]  # 过滤掉非单词和停用词
        all_words.extend(words)

    word_counts = Counter(all_words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    with open('../../brain/scifact/scifact_word_ls.tsv', 'w+') as f:
        f.write("word\tcount\n")
        for word, count in sorted_word_counts:
            f.write("{}\t{}\n".format(word, count))

def train(args):

    set_seed(args.seed)

    # --------------------load vocabulary--------------------
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # --------------------Build bert models--------------------
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)
    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained models.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    # Build classification models.
    model = BertClassifier(args, model)

    # --------------------Set the Device--------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.". format((torch.cuda.device_count())))
        model = nn.DataParallel(model)
    model = model.to(device)

    # --------------------Build knowledge Graph--------------------
    if args.kg_name == "ConceptNetWithRank":
        spo_files = args.kg_files
        kg = ConceptNetWithRank(spo_files, vocab_path=args.vocab_path, predicate=True)
    elif args.kg_name == "none":
        spo_files = []
        kg = KnowledgeGraph(spo_files=spo_files, vocab_path=args.vocab_path,predicate=True)
    else:
        spo_files = [args.kg_name]
        kg = KnowledgeGraph(spo_files=spo_files,vocab_path=args.vocab_path,predicate=True)

    # --------------------Read Data--------------------
    trainset = SciFactLabelPredictionDataset(args.corpus_path, args.train_path)
    devset = SciFactLabelPredictionDataset(args.corpus_path, args.dev_path)



    # --------------------Training Phase--------------------
    print("Start Training.")

    instances_num = len(trainset)
    batch_size = args.batch_size

    train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("\tBatch size: ", batch_size)
    print("\tThe number of training instances:", instances_num)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup, t_total=train_steps)

    patience = 0 # 未改进的epoch数
    total_loss = 0.
    result = 0.0
    best_result = 0.0

    for epoch in range(1, args.epochs_num + 1):
        model.train()
        t = tqdm(DataLoader(trainset, batch_size=args.batch_size, shuffle=True))
        for i, batch in enumerate(t):
            example = add_knowledge_worker(batch, kg, vocab, args)
            input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vms_batch = trans_to_tensor(example)

            # vms_batch = vms_batch.long()
            vms_batch = torch.LongTensor(vms_batch)
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            vms_batch = vms_batch.to(device)

            loss, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos=pos_ids_batch, vm=vms_batch)

            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                t.set_description(f'Epoch {epoch}, iter {i}, Avg loss: {round(total_loss / args.report_steps, 4)}')
                sys.stdout.flush()
                total_loss = 0.
            loss.backward()
            optimizer.step()

            if i==50:
                break
        print(f"Saving models of epoch {epoch}")
        save_model(model, f"../../outputs/scifact0429/checkpoint_*{epoch}.bin")
        train_score = evaluate(args,model, trainset, kg, vocab)
        print(f'-Epoch {epoch} train score:')
        printf(train_score)
        dev_score = evaluate(args, model, devset, kg, vocab)
        print(f'*Epoch {epoch} dev score:')
        printf(dev_score)
        result = dev_score["macro_f1"]
        print('\n')
        if result > best_result:
            best_result = result
            save_model(model, args.output_model_path)
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stopping')
            continue

    print('Final evaluation on the dev datasets')
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    printf(evaluate(args, model, devset, kg, vocab))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", type=str, default="../../models/google_model.bin", help="Path of the pretrained models.")
    parser.add_argument("--output_model_path", required=True, type=str,
                        help="Path of the output models.")
    parser.add_argument("--vocab_path", default="../../models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")

    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--dev_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)

    parser.add_argument("--config_path", default="../../models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size.")
    parser.add_argument("--seq_length", type=int, default=256,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                              "cnn", "gatedcnn", "attn", \
                                              "rcnn", "crnn", "gpt", "bilstm"], \
                        default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent models.")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Tokenizer options.
    parser.add_argument("--tokenizer", choices=["bert", "char", "word", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Word tokenizer supports online word segmentation based on jieba segmentor."
                             "Space tokenizer segments sentences into words according to space."
                        )

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--patience", type=float, default=20,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=64,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=20,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # Evaluation options.
    parser.add_argument("--mean_reciprocal_rank", action="store_true", help="Evaluation metrics for DBQA datasets.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--workers_num", type=int, default=1, help="number of process for loading datasets")
    parser.add_argument("--no_vm", action="store_true", help="Disable the visible_matrix")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    if args.kg_name == "ConceptNetWithRank":
        #args.kg_files = "../../brain/kgs/scifact_conceptnet.tsv"
        args.kg_files = [f"../../brain/scifact/conceptnet_{i}.tsv" for i in range(1,7)]

    label_set = {0,1,2}
    args.labels_num = len(label_set)

    #train(args)

    trainset = SciFactLabelPredictionDataset(args.corpus_path, args.train_path)
    devset = SciFactLabelPredictionDataset(args.corpus_path, args.dev_path)

    labels_cnt = {}
    for i in range(len(devset)):
        if devset[i]['label'] not in labels_cnt:
            labels_cnt[devset[i]['label']] = 1
        else:
            labels_cnt[devset[i]['label']] += 1

    print(labels_cnt)
    print(len(devset), len(trainset))





