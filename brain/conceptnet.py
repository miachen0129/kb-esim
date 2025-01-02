# import requests
# obj = requests.get('http://api.conceptnet.io/c/en/apple').json()
# keys = obj.keys()
#
# for i in range(len(obj['edges'])):
#     print(obj['edges'][i]["surfaceText"],obj['edges'][i]["weight"])

import nltk
import numpy as np
import requests

import brain.config as config
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize


class ConceptNet:
    def __init__(self, vocab_path, predicate=False):
        self.predicate = predicate # whether predicate
        self.vocab_path = vocab_path
        #self.tokenizer = BertTokenizer(vocab_path).tokenize
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased").tokenize
        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def lookup_up_edges(self, word, max_num):
        obj = requests.get('http://api.conceptnet.io/c/en/{}'.format(word)).json()
        edges = obj['edges']
        edges = sorted(edges, key=lambda e: e["weight"],reverse=True)
        results = set()
        cnt = 0
        for edge in edges:
            if not ('language' in edge['end'] and 'language' in edge['start']):
                continue
            if edge['end']['language'] != 'en' or edge['start']['language'] != 'en':
                continue
            if edge['rel']['label'] == 'HasProperty':
                continue
            surfaceText = edge['surfaceText']
            if edge['surfaceText'] == None:
                continue

            surfaceText = surfaceText.replace('[[',']]').split(']]')
            # subj, pred, obje = surfaceText[1], surfaceText[2].strip(), surfaceText[3]
            subj, pred, obje = edge['start']['label'], edge['rel']['label'], "_".join(edge['end']['label'].split())
            if subj != word:
                continue
            if self.predicate:
                value = pred + "[SEP] "+ obje
            else:
                value = obje
            results.add(value)

            cnt += 1
            if cnt >= max_num:
                break

        return results

    def add_knowledge_with_vm(self, sent_batch,
                              max_entities=config.MAX_ENTITIES,
                              add_pad=True,
                              max_length=128):

        split_sent_batch = [self.tokenizer(sent) for sent in sent_batch]
        know_sent_batch = []
        position_batch = []
        visible_matrix_batch = []
        seg_batch = []

        for split_sent in split_sent_batch:

            # create tree
            sent_tree = []
            pos_idx_tree = []
            abs_idx_tree = []
            pos_idx = -1
            abs_idx = -1
            abs_idx_src = []

            node_ls = []
            for token in split_sent:
                # entities = list(self.lookup_up_edges(token, max_entities))
                # sent_tree.append((token,entities))
                if token.startswith("##"):
                    # 直接推出本轮循环
                    node_ls[-1].append(token)
                    continue
                else:
                    node_ls.append([token])

            for node in node_ls:
                word = "".join([ni.replace("##","") for ni in node])
                entities = list(self.lookup_up_edges(word, max_entities))
                entities = [ent.split('[SEP]') for ent in entities]
                sent_tree.append((node, entities))
                if word in self.special_tags:
                    token_pos_idx = [pos_idx + 1]
                    token_abs_idx = [abs_idx + 1]
                else:
                    token_pos_idx = [pos_idx + i for i in range(1,len(node)+1)]
                    token_abs_idx = [abs_idx + i for i in range(1,len(node)+1)]

                abs_idx = token_abs_idx[-1]
                # 记录entities（即kg查询结果）的index
                entities_pos_idx = []
                entities_abs_idx = []
                for ent_token in entities:
                    ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent_token) + 1)]
                    entities_pos_idx.append(ent_pos_idx)
                    ent_abs_idx = [abs_idx + i for i in range(1, len(ent_token) + 1)]
                    abs_idx = ent_abs_idx[-1]
                    entities_abs_idx.append(ent_abs_idx)

                pos_idx_tree.append((token_pos_idx, entities_pos_idx))
                pos_idx = token_pos_idx[-1]
                abs_idx_tree.append((token_abs_idx, entities_abs_idx))
                abs_idx_src += token_abs_idx
            #
            # print("pos_idx_tree", pos_idx_tree)
            # print("abs_idx_tree", abs_idx_tree)
            # print("sent_tree", sent_tree)

            # Get know_sent and pos
            know_sent = []
            pos = []
            seg = []
            for i in range(len(sent_tree)):
                token = sent_tree[i][0]
                know_sent.extend(token)
                seg += [0] * len(token)
                pos += pos_idx_tree[i][0]
                for j in range(len(sent_tree[i][1])):
                    # 加入的知识
                    add_word = sent_tree[i][1][j]
                    know_sent += add_word
                    seg += [1] * len(add_word)
                    pos += pos_idx_tree[i][1][j]

            token_num = len(know_sent)
            # print("know_sent", know_sent)
            # print("pos",pos)
            # print("seg",seg)


            # Calculate visible matrix
            visible_matrix = np.zeros((token_num, token_num))
            for item in abs_idx_tree:
                src_ids = item[0]
                for id in src_ids:
                    visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                    visible_matrix[id, visible_abs_idx] = 1
                for ent in item[1]:
                    for id in ent:
                        visible_abs_idx = ent + src_ids
                        visible_matrix[id, visible_abs_idx] = 1

            src_length = len(know_sent)
            if len(know_sent) < max_length:
                pad_num = max_length - src_length
                know_sent += [config.PAD_TOKEN] * pad_num
                seg += [0] * pad_num
                pos += [max_length - 1] * pad_num
                visible_matrix = np.pad(visible_matrix, ((0, pad_num), (0, pad_num)), 'constant')  # pad 0
            else:
                know_sent = know_sent[:max_length]
                seg = seg[:max_length]
                pos = pos[:max_length]
                visible_matrix = visible_matrix[:max_length, :max_length]

            know_sent_batch.append(know_sent)
            position_batch.append(pos)
            visible_matrix_batch.append(visible_matrix)
            seg_batch.append(seg)

        return know_sent_batch, position_batch, visible_matrix_batch, seg_batch


if __name__ == "__main__":
    vocab_path = "/Users/antonchekhov/Desktop/毕业论文/basic_models/K-BERT-master/models/google_vocab.txt"
    kg = ConceptNet(vocab_path=vocab_path, predicate=True)
    sent = ['[SEP]tim cook is a ceo; a fan club [SEP] tim cook cooks']
    print(kg.add_knowledge_with_vm(sent))

