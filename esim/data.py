"""
Preprocessor and datasets definition for NLI.
"""
# Aurelien Coet, 2018.

import string
import torch
import numpy as np

from collections import Counter
# collections.Counter() 用于计数，可迭代对象出现的次数
# return 一个字典(key-value); key表示元素，value表示次数
from torch.utils.data import Dataset


class Preprocessor(object):
    """
    Preprocessor class for Natural Language Inference datasets.

    The class can be used to read NLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    用于构建词典（worddicts），将premises和hypotheses和labels转化为integer indices
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
                用于构建词典的最大词语数量，若为None，则表示全部用于计数
            stopwords: A list of words that must be ignored when building the
                worddict for a datasets. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos

    def read_data(self, filepath):
        """
        Read the premises, hypotheses and labels from some NLI datasets's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) datasets.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                pair_id = line[7]
                premise = line[1]
                hypothesis = line[2]

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(line[0])
                ids.append(pair_id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

    def read_from_tsv(self, filepath):
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")
                claim_id = line[0]
                premise = line[1]
                hypothesis = line[2]
                label = line[3]

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(label)
                ids.append(claim_id)
            #ids = [i for i in range(len(premises))]

        return {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}


    def read_tsv(self, filepath):
        labeldict = {'0':'entailment', '1':'neutral','2':'contradiction'}
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, labels = [], [], [], []

            # Translation tables to remove parentheses and punctuation from
            # strings.
            parentheses_table = str.maketrans({"(": None, ")": None})
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for i,line in enumerate(input_data):
                line = line.strip().split("\t")
                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                premise = line[1]
                hypothesis = line[2]
                id = str(i)

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(labeldict[line[0]])
                ids.append(id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}


    def read_scifact(self, dataset):

        ids, premises, hypotheses, labels = [], [], [], []
        parentheses_table = str.maketrans({"(": None, ")": None})
        punct_table = str.maketrans({key: " "
                                     for key in string.punctuation})

        if type(dataset) == dict:
            for i in range(len(dataset['label'])):
                id = str(i)
                premise = dataset['rationale'][i]
                hypothesis = dataset['claim'][i]
                label = str(int(dataset['label'][i]))

                # Remove '(' and ')' from the premises and hypotheses.
                premise = premise.translate(parentheses_table)
                hypothesis = hypothesis.translate(parentheses_table)

                if self.lowercase:
                    premise = premise.lower()
                    hypothesis = hypothesis.lower()

                if self.ignore_punctuation:
                    premise = premise.translate(punct_table)
                    hypothesis = hypothesis.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                premises.append([w for w in premise.rstrip().split()
                                 if w not in self.stopwords])
                hypotheses.append([w for w in hypothesis.rstrip().split()
                                   if w not in self.stopwords])
                labels.append(label)
                ids.append(id)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels}

        for i in range(len(dataset)):
            id = str(i)
            premise = dataset[i]['rationale']
            hypothesis = dataset[i]['claim']
            label = str(dataset[i]['label'])

            # Remove '(' and ')' from the premises and hypotheses.
            premise = premise.translate(parentheses_table)
            hypothesis = hypothesis.translate(parentheses_table)

            if self.lowercase:
                premise = premise.lower()
                hypothesis = hypothesis.lower()

            if self.ignore_punctuation:
                premise = premise.translate(punct_table)
                hypothesis = hypothesis.translate(punct_table)

            # Each premise and hypothesis is split into a list of words.
            premises.append([w for w in premise.rstrip().split()
                             if w not in self.stopwords])
            hypotheses.append([w for w in hypothesis.rstrip().split()
                               if w not in self.stopwords])
            labels.append(label)
            ids.append(id)

        return {"ids": ids,
                "premises": premises,
                "hypotheses": hypotheses,
                "labels": labels}





    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some datasets. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI datasets, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        # 将premises和hypothesis中的句子全部加入words list中
        [words.extend(sentence) for sentence in data["premises"]]
        [words.extend(sentence) for sentence in data["hypotheses"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        # 将用于padding的、不计数的、BOS和EOS进行特别的标注
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        # offset表示worddict中不表示实际有用词语的个数
        # 0是padding，1是oov，2是bos，3是eos
        # 未保证后加进去的词语indices不与offset中重合，后续indice=frequency+offset
        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        # 选出出现频率前num_words的词语，加入对象的worddict中
        # key为word；value为 frequency + offset
        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the premises and hypotheses of a datasets, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of premises, hypotheses
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed premises, hypotheses and
            labels.
        """
        transformed_data = {"ids": [],
                            "premises": [],
                            "hypotheses": [],
                            "labels": []}

        for i, premise in enumerate(data["premises"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                print(label, self.labeldict)
                continue


            transformed_data["ids"].append(data["ids"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(premise)
            transformed_data["premises"].append(indices)

            indices = self.words_to_indices(data["hypotheses"][i])
            transformed_data["hypotheses"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        # 对没有embedding的词语进行计数
        # 未在worddict中的词语
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix


class NLIDataset(Dataset):
    """
    Dataset class for Natural Language Inference datasets.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_premise_length=None,
                 max_hypothesis_length=None):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some datasets.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_premise_length: An integer indicating the maximum length
                accepted for the sequences in the premises. If set to None,
                the length of the longest premise in 'data' is used.
                Defaults to None.
            max_hypothesis_length: An integer indicating the maximum length
                accepted for the sequences in the hypotheses. If set to None,
                the length of the longest hypothesis in 'data' is used.
                Defaults to None.
        """
        self.premises_lengths = [len(seq) for seq in data["premises"]]
        self.max_premise_length = max_premise_length
        if self.max_premise_length is None:
            self.max_premise_length = max(self.premises_lengths)

        self.hypotheses_lengths = [len(seq) for seq in data["hypotheses"]]
        self.max_hypothesis_length = max_hypothesis_length
        if self.max_hypothesis_length is None:
            self.max_hypothesis_length = max(self.hypotheses_lengths)

        self.num_sequences = len(data["premises"])

        self.data = {"ids": [],
                     "premises": torch.ones((self.num_sequences,
                                             self.max_premise_length),
                                            dtype=torch.long) * padding_idx,
                     "hypotheses": torch.ones((self.num_sequences,
                                               self.max_hypothesis_length),
                                              dtype=torch.long) * padding_idx,
                     "labels": torch.tensor(data["labels"], dtype=torch.long)}

        for i, premise in enumerate(data["premises"]):
            self.data["ids"].append(data["ids"][i])
            end = min(len(premise), self.max_premise_length)
            self.data["premises"][i][:end] = torch.tensor(premise[:end])

            hypothesis = data["hypotheses"][i]
            end = min(len(hypothesis), self.max_hypothesis_length)
            self.data["hypotheses"][i][:end] = torch.tensor(hypothesis[:end])

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.premises_lengths[index],
                                      self.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.hypotheses_lengths[index],
                                         self.max_hypothesis_length),
                "label": self.data["labels"][index]}

if __name__ == "__main__":
    test_file_esim = "./datasets/snli/snli_1.0_test.txt"

    data = preprocessor.read_data(test_file_esim)
    premises, hypotheses, labels = data['premises'], data['hypotheses'], data['labels']