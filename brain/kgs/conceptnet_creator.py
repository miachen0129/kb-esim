import requests
from collections import Counter
import pickle
import os
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset

import fnmatch


class SciTailDataset(Dataset):
    def __init__(self, file : str):
        self.samples = []

        label_encodings = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        with open(file, 'r', encoding='utf8') as input_data:
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                # Ignore sentences that have no gold label.
                if line[0] == "-":
                    continue

                label = line[0]
                premise = line[5]
                hypothesis = line[6]

                self.samples.append({'claim': hypothesis,'rationale':premise, 'label':label_encodings[label]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def create_conceptnet(words, targetdir, no):
    results = []
    f = open(os.path.join(targetdir, 'conceptnet_{}.tsv'.format(no)), 'a')
    f.write("{}\t{}\t{}\n".format("subject", "predicate", "object"))
    for i,word in tqdm(enumerate(words),desc="Processing", total=len(words)):
        if i%100 == 0:
            print("reach {} words out of {} words".format(i,len(words)))
        try:
            obj = requests.get('http://api.conceptnet.io/c/en/{}'.format(word)).json()
            edges = obj['edges']
            edges = sorted(edges, key=lambda e: e["weight"], reverse=True)
            cnt = 0
            for edge in edges:
                if cnt>3:
                    break
                if not ('language' in edge['end'] and 'language' in edge['start']):
                    continue
                if edge['end']['language'] != 'en' or edge['start']['language'] != 'en':
                    continue
                if edge['rel']['label'] in ['HasProperty','AtLocation','Desires','MotivatedByGoal','CauseDesire']:
                    continue
                surfaceText = edge['surfaceText']
                if surfaceText == None:
                    continue

                # surfaceText = surfaceText.replace('[[', ']]').split(']]')
                # subj, pred, obje = surfaceText[1], surfaceText[2].strip(), surfaceText[3]
                subj, pred, obje = edge['start']['label'].lower(), edge['rel']['label'], "_".join(edge['end']['label'].split())
                if word not in subj:
                    #print("{} not in {}".format(word, subj))
                    continue
                subj = word
                results.append((subj,pred,obje))
                cnt += 1

                f.write("{}\t{}\t{}\n".format(subj, pred, obje))
        except:
            print("skipping line {}".format(i))

    f.close()
    # print("starting to save data!")
    # results = list(results)
    # with open(os.path.join(targetdir,'conceptnet_{}.tsv'.format(no)), 'w') as f:
    #     f.write("{}\t{}\t{}\n".format("subject","predicate","object"))
    #     for result in results:
    #         f.write("{}\t{}\t{}\n".format(result[0], result[1], result[2]))
    print("successfully saved data!")

def create_word_ls(inputdir):
    train_file = os.path.join(inputdir, "scitail_train.txt")
    test_file = os.path.join(inputdir, "scitail_test.txt")
    dev_file = os.path.join(inputdir, "scitail_dev.txt")

    trainset = SciTailDataset(train_file)
    testset = SciTailDataset(test_file)
    devset = SciTailDataset(dev_file)

    sentences = []
    all_words = []
    stop_words = set(stopwords.words('english'))

    for line in trainset:
        claim = line['claim']
        rationale = line['rationale']
        if claim not in sentences:
            sentences.append(claim)
            words = nltk.word_tokenize(claim.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)
        if rationale not in sentences:
            sentences.append(rationale)
            words = nltk.word_tokenize(rationale.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)

    for line in testset:
        claim = line['claim']
        rationale = line['rationale']
        if claim not in sentences:
            sentences.append(claim)
            words = nltk.word_tokenize(claim.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)
        if rationale not in sentences:
            sentences.append(rationale)
            words = nltk.word_tokenize(rationale.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)

    for line in devset:
        claim = line['claim']
        rationale = line['rationale']
        if claim not in sentences:
            sentences.append(claim)
            words = nltk.word_tokenize(claim.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)
        if rationale not in sentences:
            sentences.append(rationale)
            words = nltk.word_tokenize(rationale.lower())
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)

    # 统计词频并排序
    word_counts = Counter(all_words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    with open('../../datasets/scitail/word_ls.tsv','w') as f:
        f.write("word\tcount\n")
        for word, count in sorted_word_counts:
            f.write("{}\t{}\n".format(word, count))

def count_words():
    # 加载停用词列表
    stop_words = set(stopwords.words('english'))
    wml = WordNetLemmatizer()
    all_words = []
    with open('../../datasets/scifact/kbert_rte_train.tsv', encoding='utf8') as f:
        for line in f.readlines():
            # 将文本转换为单词列表并去除停用词
            words = nltk.word_tokenize(line.lower())  # 将文本转换为小写，并将其拆分为单词列表
            words = [word for word in words if word.isalpha() and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)
    with open('../../datasets/scifact/kbert_rte_dev.tsv', encoding='utf8') as f:
        for line in f.readlines():
            # 将文本转换为单词列表并去除停用词
            words = nltk.word_tokenize(line.lower())  # 将文本转换为小写，并将其拆分为单词列表
            words = [word for word in words if word.isalnum()and len(word)>1 and word not in stop_words][1:]  # 过滤掉非单词和停用词
            all_words.extend(words)
    # 统计词频并排序
    word_counts = Counter(all_words)
    sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

    with open('../../datasets/scifact/word_ls.tsv','w') as f:
        f.write("word\tcount\n")
        for word, count in sorted_word_counts:
            f.write("{}\t{}\n".format(word, count))




def main():
    words = []
    cnt = 0
    targetdir = '../../brain/scifact'
    with open('../../brain/scifact_word_ls.tsv', encoding='utf8') as f:
        for line in f.readlines():
            word = line.strip().split('\t')[0]
            words.append(word)
            cnt+=1
            if cnt == 10000:
                break

    create_conceptnet(words[-2500:], targetdir, 6)




if __name__ == '__main__':
    # word = 'cranium'
    # obj = requests.get('http://api.conceptnet.io/c/en/{}'.format(word)).json()
    # edges = obj['edges']
    # edges = sorted(edges, key=lambda e: e["weight"], reverse=True)
    # cnt = 0
    # results = []
    # for edge in edges:
    #     if cnt > 3:
    #         break
    #     if not ('language' in edge['end'] and 'language' in edge['start']):
    #         continue
    #     if edge['end']['language'] != 'en' or edge['start']['language'] != 'en':
    #         continue
    #     if edge['rel']['label'] in ['HasProperty', 'AtLocation', 'Desires', 'MotivatedByGoal', 'CauseDesire']:
    #         continue
    #     surfaceText = edge['surfaceText']
    #     if surfaceText == None:
    #         continue
    #
    #     # surfaceText = surfaceText.replace('[[', ']]').split(']]')
    #     # subj, pred, obje = surfaceText[1], surfaceText[2].strip(), surfaceText[3]
    #     subj, pred, obje = edge['start']['label'].lower(), edge['rel']['label'], "_".join(edge['end']['label'].split())
    #     if word not in subj:
    #         # print("{} not in {}".format(word, subj))
    #         continue
    #     subj = word
    #     #results.append((subj, pred, obje))
    #     print((subj, pred, obje))

    create_conceptnet(['cranium','brain'],'../scitail',2)