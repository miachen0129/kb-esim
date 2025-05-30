
import pickle
import torch
from esim.data import NLIDataset, Preprocessor
from esim.model import ESIM
import os
import argparse
import json


def predict_by_esim(model, preprocessor, ids, premises, hypotheses, labels):
    """
    :param model: trained ESIM models
    :param preprocessor: 预处理工具
    :param ids: id列表，对应p-h组的唯一索引
    :param premises: 前提
    :param hypotheses: 假设
    :param labels: 标签（必须为'entailment','contradiction','neutral'）
    """
    labeldict= {'entailment':0, 'contradiction':2,'neutral':1}
    #data = {"ids": ids, "premises": premises, "hypotheses": hypotheses, "labels": labels}
    data = preprocessor.read_data("datasets/snli/snli_1.0_test.txt")
    print(data)
    transformed_data = preprocessor.transform_to_indices(data)
    dataset = NLIDataset(transformed_data)

    model.eval()
    device = model.device

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        premises = dataset.data['premises'].to(device)
        premises_lengths = torch.tensor(dataset.premises_lengths)
        hypotheses = dataset.data['hypotheses'].to(device)
        hypotheses_lengths = torch.tensor(dataset.hypotheses_lengths)

        _, probs = model(premises,
                        premises_lengths,
                        hypotheses,
                        hypotheses_lengths)

        out_classes = probs.max(dim=1).indices

        correct = torch.sum(out_classes == dataset.data["labels"]).item()
        print(correct/len(ids))

        return probs, out_classes
def load_esim_params(device,
                     esim_path,
                     default_config="config/preprocessing/snli_preprocessing.json",
                     wordict_path="datasets/preprocessed/worddict.pkl"):

    # ------------------Load The Model------------------
    checkpoint = torch.load(esim_path)
    ## Retrieving models parameters from checkpoint.
    vocab_size = checkpoint["models"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["models"]['_word_embedding.weight'].size(1)
    hidden_size = checkpoint["models"]["_projection.0.weight"].size(0)
    num_classes = checkpoint["models"]["_classification.4.weight"].size(0)
    ## Create and then load the models
    model = ESIM(vocab_size,
                embedding_dim,
                hidden_size,
                num_classes=num_classes,
                device=device).to(device)

    model.load_state_dict(checkpoint["models"])

    # ----------------Load the Preprocessor----------------
    ## load the preprocessor configs from json file
    parser = argparse.ArgumentParser(description="Preprocess the SNLI datasets")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing SciFact datasets"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    ## Create the preprocessor according to the configs
    preprocessor = Preprocessor( lowercase=config["lowercase"],
                                 ignore_punctuation=config["ignore_punctuation"],
                                 num_words=config["num_words"],
                                 stopwords=config["stopwords"],
                                 labeldict=config["labeldict"],
                                 bos=config["bos"],
                                 eos=config["eos"]
                                )
    ## Load the word dict
    with open(wordict_path,"rb") as file:
        worddict = pickle.load(file)
    preprocessor.worddict = worddict

    return model, preprocessor


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    esim_path = "../../outputs/pretrained_esim.pth.tar"
    default_config = "config/preprocessing/snli_preprocessing.json"
    wordict_path = "../../datasets/preprocessed/worddict.pkl"

    model, preprocessor = load_esim_params(device, esim_path, default_config, wordict_path)
    #print(preprocessor.worddict)

    # load the data
    reverse_labeldict = {'0':'entailment','1':'neutral','2':'contradiction'}
    test_file = "../../datasets/snli/test.tsv"
    ids, premises, hypotheses, labels = [],[],[],[]
    with open(test_file, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            else:
                sents = line.strip().split("\t")
                ids.append(str(i))
                premises.append(sents[1])
                hypotheses.append(sents[2])
                labels.append(reverse_labeldict[sents[0]])


    probs = predict_by_esim(model, preprocessor, ids[:100], premises[:100], hypotheses[:100], labels[:100])
    print(probs.size())
    return ids, probs




if __name__ == '__main__':
    main()