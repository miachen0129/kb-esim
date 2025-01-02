from multiprocessing import Pool

from brain import KnowledgeGraph
from brain.conceptnet import ConceptNet
from uer.utils.constants import *
import sys
import os
import fnmatch
import pickle
from uer.utils.vocab import Vocab


def add_knowledge_worker(params):
    """
    构建注入知识的dataset
    :param p_id:
    :param sentences: 每一行的列表
    :param columns: 列名对应的排号的dict
    :param kg: knowledge graph
    :param vocab: vocabulary
    :param seg_length:
    """
    p_id, sentences, columns, kg, vocab, seg_length = params
    sentences_num = len(sentences)
    dataset = []
    for line_id, line in enumerate(sentences):
        if line_id % 1000 == 0:
            print("Progress of process {}: {}/{}".format(p_id, line_id, sentences_num))
            sys.stdout.flush() # 清空缓冲区
        line = line.strip().split('\t')

        # try:
        if len(line) == 3:
                # label必须为int
                label = int(line[columns["label"]])
                text = " ".join([CLS_TOKEN,line[columns["text_a"]],SEP_TOKEN,line[columns["text_b"]],SEP_TOKEN])

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=seg_length)
                # ===========为什么要选第一个？？？？============
                tokens = tokens[0]
                pos = pos[0]
                vm = vm[0].astype("bool") #转换成bool形式

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

        elif len(line) == 4:  # for claim verification
                cid = int(line[columns["cid"]])
                label = int(line[columns["label"]])
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                text = " ".join([CLS_TOKEN,line[columns["text_a"]],SEP_TOKEN,line[columns["text_b"]],SEP_TOKEN])

                tokens, pos, vm, _ = kg.add_knowledge_with_vm([text], add_pad=True, max_length=seg_length)
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

                # 保留cid，方便后面再检索
                dataset.append((token_ids, label, mask, pos, vm, cid))

        else:
                print("Skipping line")
        # except:
        #     print("Error in line {}: {}".format(line_id, line))


    return dataset

def read_dataset(path, columns, kg, vocab, seg_length, workers_num=1):
    print("Loading sentences from {}".format(path))
    sentences = []  # 所有行
    with open(path, mode='r', encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            sentences.append(line)
    sentence_num = len(sentences)  # 行数，即样本数

    # 看不懂这里的workers是啥，但大概指用多少个过程注入知识
    print("There are {} sentence in total. We use {} processes to inject knowledge into sentences.".format(sentence_num,
                                                                                                           workers_num))
    if workers_num > 1:
        params = []
        sentence_per_block = int(sentence_num / workers_num) + 1
        for i in range(workers_num):
            params.append(
                (i, sentences[i * sentence_per_block: (i + 1) * sentence_per_block], columns, kg, vocab, seg_length))
        pool = Pool(workers_num)
        res = pool.map(add_knowledge_worker, params)
        pool.close()
        pool.join()
        dataset = [sample for block in res for sample in block]
    else:
        params = (0, sentences, columns, kg, vocab, seg_length)
        dataset = add_knowledge_worker(params)

    return dataset

def main(inputdir,
         targetdir,
         kg_name,
         vocab_path,
         seg_length,
         workers_num,
         ):

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "train.tsv"):
            train_file = file
        elif fnmatch.fnmatch(file, "dev.tsv"):
            dev_file = file
        elif fnmatch.fnmatch(file, "test.tsv"):
            test_file = file

    # Count the number of labels.
    labels_set = set()
    columns = {}
    with open(os.path.join(inputdir,train_file), mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            try:
                line = line.strip().split("\t")
                if line_id == 0:
                    for i, column_name in enumerate(line):
                        columns[column_name] = i
                    continue
                label = int(line[columns["label"]])
                labels_set.add(label)
            except:
                pass

    # load vocabulary
    print("Loading Vocabulary...")
    vocab = Vocab()
    vocab.load(vocab_path)

    # Build knowledge graph
    print("Building knowledge graph {}..".format(kg_name))
    if kg_name == 'ConceptNet':
        kg = ConceptNet(predicate=True, vocab_path='vocab_path')
    else:
        if kg_name == 'none':
            spo_files = []
        else:
            spo_files = [kg_name]
        kg = KnowledgeGraph(spo_files, vocab_path=vocab_path,predicate=True)

    # add knowledge to training dataset and save
    print("Injecting knowledge to train dataset..")
    transformed_data = read_dataset(path=os.path.join(inputdir, train_file),
                            columns=columns,
                            vocab=vocab,
                            kg=kg,
                            seg_length=seg_length,
                            workers_num=workers_num
                            )
    print(transformed_data)
    print("Saving train dataset..")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    print("Injecting knowledge to dev dataset..")
    # add knowledge to dev dataset and save
    transformed_data = read_dataset(path=os.path.join(inputdir, dev_file),
                                    columns=columns,
                                    vocab=vocab,
                                    kg=kg,
                                    seg_length=seg_length,
                                    workers_num=workers_num
                                    )
    print("Saving dev dataset..")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)

    # add knowledge to test dataset and save
    print("Injecting knowledge to test dataset..")
    transformed_data = read_dataset(path=os.path.join(inputdir, test_file),
                                    columns=columns,
                                    vocab=vocab,
                                    kg=kg,
                                    seg_length=seg_length,
                                    workers_num=workers_num
                                    )
    print("Saving test dataset..")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_data, pkl_file)



if __name__ == "__main__":
    inputdir = "./datasets/snli"
    targetdir = "./datasets/preprocessed/snli_hownet"
    kg_name = 'HowNet'
    vocab_path = "../../models/google_vocab.txt"
    seg_length = 256
    workers_num = 1
    main(inputdir,targetdir,kg_name,vocab_path,seg_length,workers_num)
