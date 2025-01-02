import torch.nn

from scripts.testing.ensemble_model import *
def label_aggr(avg_prob):
    # print(avg_prob)
    # bias = 1 / torch.cosine_similarity(avg_prob, torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=float))
    # weighted_probs = torch.sum(torch.mul(avg_prob, bias.view(avg_prob.shape[0], 1)), dim=0)
    # avg_prob = weighted_probs
    # #avg_prob = avg_prob.mean(dim=0)
    # print(avg_prob)
    # if avg_prob[0] > avg_prob[2]:
    #     final_predict = 0
    # elif avg_prob[0] < avg_prob[2]:
    #     final_predict = 2
    # else:
    #     final_predict = 1
    bias = 1 / torch.cosine_similarity(avg_prob, torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=float))
    class_cnt = torch.zeros(3)
    avg_pred = torch.argmax(avg_prob, dim=1)
    for i in range(len(avg_pred)):
        predl = int(avg_pred[i])
        class_cnt[predl] += bias[i]

    class_cnt = class_cnt / torch.sum(class_cnt)

    if class_cnt[0] > class_cnt[2]:
        final_predict = 0
    elif class_cnt[2] > class_cnt[0]:
        final_predict = 2
    else:
        final_predict = 1
    return  class_cnt

def claim_verification():
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------Kbert preparation--------------------
    print("start preparing kbert")
    columns = {"text_a": 1, "text_b": 2, "label": 0}
    kbert_path = "../../outputs/scifact_models/kbert2.bin"

    kbert, vocab, kg, args = load_kbert_params(device, kbert_path, columns, kgname='ConceptNetWithRank', isRank=True)
    kbert = kbert.to(device)

    # --------------------ESIM preparation---------------------
    print("start preparing esim models")
    esim_path = "../../outputs/scifact_models/best2.pth.tar"
    default_config = "config/preprocessing/scifact_preprocessing.json"
    wordict_path = "../../outputs/scifact_models/worddict.pkl"
    esim, preprocessor = load_esim_params(device, esim_path, default_config, wordict_path)
    reverse_labeldict = {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}
    label_dict = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    # ----------Preparing Tesing data----------
    print("start preparing data")
    test_file = "../../datasets/scifact_2/rte_dev.tsv"
    data = preprocessor.read_from_tsv(test_file)
    cids, premises, hypotheses, labels = data["ids"], data['premises'], data['hypotheses'], data['labels']
    ids, sentences, label_ids = [], [], []
    for i in range(len(premises)):
        ids.append(str(i + 1))
        sentences.append([str(label_dict[labels[i]]), " ".join(premises[i]), " ".join(hypotheses[i])])
        label_ids.append(label_dict[labels[i]])

    def instance_loader(cids, ids, premises, hypotheses, sentences, labels, label_ids):
        instance_dict = dict()
        for i in range(len(premises)):
            if cids[i] not in instance_dict:
                instance_dict[cids[i]] = []
            instance_dict[cids[i]].append([ids[i], premises[i], hypotheses[i], sentences[i], labels[i], label_ids[i]])

        for cid in instance_dict:
            instances_batch = instance_dict[cid]
            ids_batch = [item[0] for item in instances_batch]
            premises_batch = [item[1] for item in instances_batch]
            hypotheses_batch = [item[2] for item in instances_batch]
            sentences_batch = [item[3] for item in instances_batch]
            labels_batch = [item[4] for item in instances_batch]
            label_ids_batch = [item[5] for item in instances_batch]
            if 'entailment' in labels_batch:
                yield 0,ids_batch, premises_batch, hypotheses_batch, sentences_batch, labels_batch, label_ids_batch
            elif 'contradiction' in labels_batch:
                yield 2, ids_batch, premises_batch, hypotheses_batch,sentences_batch, labels_batch, label_ids_batch
            else:
                yield 1,ids_batch, premises_batch, hypotheses_batch, sentences_batch, labels_batch, label_ids_batch

    correct = 0
    oracle_correct = 0
    kbert_correct = 0
    esim_correct = 0
    total_cnt = len(list(set(cids)))
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)

    for i, (claim_label, ids_batch, premises_batch, hypotheses_batch, sentences_batch, labels_batch, label_ids_batch) in \
            enumerate(instance_loader(cids, ids, premises, hypotheses, sentences, labels, label_ids)):
        print("-"*40)
        print("testing claim {}, claim label is {} ".format(i, claim_label))
        print("there are premises: {}".format(premises_batch))
        if ["[NOINFO]"] in premises_batch:
            print("there is no information available")
            final_predict = 1
            final2 = 1
            final1 = 1
        else:
            kb_dataset = read_dataset(sentences_batch, columns, kg, vocab, args, workers_num=args.workers_num)
            kb_ids = [item[0] for item in kb_dataset]
            kb_dataset = [item[1:] for item in kb_dataset]
            # predict by kbert
            logits = predict_by_kbert(kb_dataset, kbert)
            # predict by esim
            probs = predict_by_esim(esim, preprocessor, ids_batch, premises_batch, hypotheses_batch, labels_batch)
            # 求平均
            if logits.size() == probs.size():
                avg_prob = (logits+probs)/2
            else:
                avg_prob = torch.zeros(probs.size())
                for j, kb_id in enumerate(kb_ids):
                    avg_prob[int(kb_id)-1] = (logits[j]+probs[int(kb_id)-1])/2
            #avg_prob = avg_prob.mean(dim=0)
            # final1 = label_aggr(logits)
            # final2 = label_aggr(probs)
            # final_predict = label_aggr(avg_prob)
            esim_probs = torch.mean(probs, dim=0)
            print(esim_probs)
            kbert_probs = label_aggr(logits)
            print(esim_probs)
            class_cnt = (esim_probs+kbert_probs)/2

            print(class_cnt)
            if class_cnt[0] > class_cnt[2]:
                final_predict = 0
            elif class_cnt[2] > class_cnt[0]:
                final_predict = 2
            else:
                final_predict = 1

        print(final_predict, claim_label)
        if final_predict == claim_label:
            correct += 1
        # if final2 == claim_label:
        #     esim_correct += 1
        # if final1 == claim_label:
        #     kbert_correct += 1
        # if final1 == claim_label or final2 == claim_label:
        #     oracle_correct += 1


        confusion[final_predict, claim_label] += 1

    print(confusion)
    # for i in range(confusion.size()[0]):
    #     p = confusion[i, i].item() / confusion[i, :].sum().item()
    #     r = confusion[i, i].item() / confusion[:, i].sum().item()
    #     f1 = 2 * p * r / (p + r)
    #     if i == 1:
    #        label_1_f1 = f1
        #print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
    print("Accuracy:{}".format(correct / total_cnt))
    print("ESIM Accuracy:{}".format(esim_correct/total_cnt))
    print("KBERT Accuracy:{}".format(kbert_correct/total_cnt))
    print("ORACLE Accuracy:{}".format(oracle_correct/total_cnt))


if __name__ == '__main__':
    claim_verification()