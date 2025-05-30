import brain.config as config
import numpy as np
from uer.utils.tokenizer import BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class ConceptNetWithRank(object):
    """
    spo_files - list of Path of *.spo files, or default kg name. e.g., ['HowNet']
    """

    def __init__(self, kg_files, vocab_path, predicate=False):
        # 谓词
        self.predicate = predicate
        # config.KGS 是各个KG的spo文件路径
        self.kg_files = kg_files
        self.tokenizer = BertTokenizer(vocab_path=vocab_path).tokenize
        self.lookup_table = self._create_lookup_table()
        self.segment_vocab = list(self.lookup_table.keys()) + config.NEVER_SPLIT_TAG
        #self.tokenizer = pkuseg.pkuseg(model_name="default", postag=False, user_dict=self.segment_vocab)

        self.special_tags = set(config.NEVER_SPLIT_TAG)

    def _create_lookup_table(self):
        """
        应用于knowledge layer中的第一步，K_Query部分
        """
        lookup_table = {}
        for kg_path in self.kg_files:
            print("[KnowledgeGraph] Loading spo from {}".format(kg_path))
            with open(kg_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # KG中所有元素被转构成为(w_i,r_j,w_k)元组，w为实体，r为关系。
                        # 识别spo文件中的每一行，分别存入三元组中。
                        # 无法通过此方法识别的就被列入bad spo
                        subj, pred, obje = line.strip().split("\t")
                        rpred = pred[0].lower()
                        for cha in pred[1:]:
                            if cha.islower():
                                rpred += cha
                            else:
                                rpred += "_{}".format(cha.lower())
                    except:
                        print("[KnowledgeGraph] Bad spo:", line)
                    if self.predicate:
                        value = "[SEP]".join([rpred,obje])
                    else:
                        value = obje
                    if subj in lookup_table.keys():
                        lookup_table[subj].add(value)
                    else:
                        lookup_table[subj] = set([value])
        return lookup_table

    def K_Query(self, split_sent, word, max_entities=config.MAX_ENTITIES):
        entities = list(self.lookup_table.get(word, []))
        candidates = []
        for entity in entities:
            entity = entity.split('[SEP]')
            candidate = []
            [candidate.extend(ent.split("_")) for ent in entity]
            candidates.append(candidate)
        # using bleu scores to find most similar interpretations from KG
        scores = [(i, sentence_bleu(" ".join(split_sent),
                                            " ".join(candidate),
                                            smoothing_function=SmoothingFunction().method1))
                  for i,candidate in enumerate(candidates)]

        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        results = [entities[0] for item in sorted_scores]

        return results


    def add_knowledge_with_vm(self, sent_batch, max_entities=config.MAX_ENTITIES, add_pad=True,
                                       max_length=128):
        """
        input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
        return: know_sent_batch - list of sentences with entites embedding
                position_batch - list of position index of each character.
                visible_matrix_batch - list of visible matrixs
                seg_batch - list of segment tags
        """
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
                if token.startswith("##"):
                    # 直接推出本轮循环
                    node_ls[-1].append(token)
                    continue
                else:
                    node_ls.append([token])

            for node in node_ls:
                word = "".join([ni.replace("##","") for ni in node])
                entities = self.K_Query(split_sent, word, max_entities)
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

if __name__ == '__main__':
    kg_files = ["scitail/conceptnet_1.tsv","scitail/conceptnet_2.tsv"]
    vocab_path = "../models/google_vocab.txt"
    kg = ConceptNetWithRank(kg_files, vocab_path=vocab_path, predicate=True)
    sent = ['[CLS] The cranium protects the brain. [SEP]']
    # know_sent_batch, position_batch, visible_matrix_batch, seg_batch = kg.add_knowledge_with_vm(sent)
    # print(know_sent_batch)
    # print(position_batch)
    # print(visible_matrix_batch)
    # print(seg_batch)
    print(kg.K_Query(sent, 'brain'))
