import json
import os
import fnmatch
import random
import jsonlines
from torch.utils.data import Dataset


class SciFactLabelPredictionDataset(Dataset):
    def __init__(self, corpus: str, claims: str):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        label_encodings = {'CONTRADICT': 2, 'NOT_ENOUGH_INFO': 1, 'SUPPORT': 0}

        for claim in jsonlines.open(claims):
            if claim['evidence']:
                for doc_id, evidence_sets in claim['evidence'].items():
                    doc = corpus[int(doc_id)]

                    # Add individual evidence set as samples:
                    for evidence_set in evidence_sets:
                        rationale = [doc['abstract'][i].strip() for i in evidence_set['sentences']]
                        self.samples.append({
                            'claim': claim['claim'],
                            'rationale': ' '.join(rationale),
                            'label': label_encodings[evidence_set['label']]
                        })

                    # Add all evidence sets as positive samples
                    rationale_idx = {s for es in evidence_sets for s in es['sentences']}
                    rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(rationale_sentences),
                        'label': label_encodings[evidence_sets[0]['label']]  # directly use the first evidence set label
                        # because currently all evidence sets have
                        # the same label
                    })

                    # Add negative samples
                    non_rationale_idx = set(range(len(doc['abstract']))) - rationale_idx
                    non_rationale_idx = random.sample(non_rationale_idx,
                                                      k=min(random.randint(1, 2), len(non_rationale_idx)))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in sorted(list(non_rationale_idx))]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })
            else:
                # Add negative samples
                for doc_id in claim['cited_doc_ids']:
                    doc = corpus[int(doc_id)]
                    non_rationale_idx = random.sample(range(len(doc['abstract'])), k=random.randint(1, 2))
                    non_rationale_sentences = [doc['abstract'][i].strip() for i in non_rationale_idx]
                    self.samples.append({
                        'claim': claim['claim'],
                        'rationale': ' '.join(non_rationale_sentences),
                        'label': label_encodings['NOT_ENOUGH_INFO']
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def dataset(inputdir, iftrain=True):
    # Retrieve the train, dev and test data files from the datasets directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_train.jsonl"):
            train_file = file
        elif fnmatch.fnmatch(file, "*_dev.jsonl"):
            dev_file = file
        elif fnmatch.fnmatch(file, "*_test.jsonl"):
            test_file = file

    cls0_cnt, cls1_cnt, cls2_cnt = 0, 1, 2
    if iftrain:
        path = train_file
    else:
        path = dev_file
    with open(os.path.join(inputdir, path), encoding='utf8') as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            evidence = data['evidence']
            if evidence == {}:
                cls1_cnt += 1
                continue
            for key in evidence.keys():
                label = evidence[key][0]['label']
                if label == 'SUPPORT':
                    cls0_cnt += 1
                elif label == 'CONTRADICT':
                    cls2_cnt += 1
                else:
                    print(label)

    print(cls0_cnt, cls1_cnt, cls2_cnt)

def printf(data):
    keys = data.keys()
    for key in keys:
        print(key,':',data[key])

def withcitance(inputdir):
    # for claim with citances
    path = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_with_citances.jsonl"):
            path = os.path.join(inputdir, file)
    cnt, ncnt = 0,0
    with open(path, 'r', encoding='utf8') as f:
        for i,line in enumerate(f):
            data = json.loads(line)
            #printf(data)
            for claim in data['claims']:
                if claim['is_negation'] == False:
                    ncnt += 1
                elif claim['is_negation'] == True:
                    cnt += 1
                else:
                    print(claim)
    print(cnt, ncnt)

def main():
    inputdir = "../datasets/meta/sciFact"
    corpus = ""
    train = ""
    dev = ""
    test = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*corpus.jsonl"):
            corpus = os.path.join(inputdir, file)
        if fnmatch.fnmatch(file, "*_train.jsonl"):
            train = os.path.join(inputdir, file)
        if fnmatch.fnmatch(file, "*_test.jsonl"):
            test = os.path.join(inputdir, file)
        if fnmatch.fnmatch(file, "*_dev.jsonl"):
            dev = os.path.join(inputdir, file)

    dataset = SciFactLabelPredictionDataset(corpus, dev)
    class_cnt = {}
    for i in range(len(dataset)):
        data = dataset[i]
        if i == 0:
            printf(data)
        if data['label'] in class_cnt:
            class_cnt[data['label']] += 1
        else:
            class_cnt[data['label']] = 1

    targetdir = "../datasets/reconstructed/scifact"
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    with open(os.path.join(targetdir, "dev.tsv"),'w', encoding='utf8') as f:
        f.write("\t".join(["label", "text_a","text_b"])+"\n")
        for i in range(len(dataset)):
            data = dataset[i]
            text_a = data['rationale']
            text_b = data['claim']
            label = str(data['label'])
            f.write("\t".join([label, text_a,text_b])+"\n")

    print(class_cnt)
if __name__ == "__main__":
    inputdir = "../../datasets/meta/sciFact"
    #datasets(inputdir, True)
    main()
