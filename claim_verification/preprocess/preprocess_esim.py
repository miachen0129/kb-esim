import os
import argparse
import pickle
import json
import fnmatch

from claim_verification.utils.Datasets import SciFactLabelPredictionDataset
from esim.data import Preprocessor

def preprocess(inputdir,
               embeddings_file,
               targetdir,
               lowercase=False,
               ignore_punctuation=False,
               num_words=None,
               stopwords=[],
               labeldict={},
               bos=None,
               eos=None):

    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the datasets directory.
    train_file = ""
    dev_file = ""
    corpus_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "*_train.jsonl"):
            train_file = os.path.join(inputdir,file)
        elif fnmatch.fnmatch(file, "*_dev.jsonl"):
            dev_file = os.path.join(inputdir,file)
        elif fnmatch.fnmatch(file, "corpus.jsonl"):
            corpus_file = os.path.join(inputdir,file)

    print(train_file, dev_file, corpus_file)

    trainset = SciFactLabelPredictionDataset(corpus_file, train_file)
    devset = SciFactLabelPredictionDataset(corpus_file, dev_file)

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict,
                                bos=bos,
                                eos=eos)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    train_data = preprocessor.read_scifact(trainset)

    print("\t* Computing worddict and saving it...")
    preprocessor.build_worddict(train_data)
    with open(os.path.join(targetdir, "worddict.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor.worddict, pkl_file)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_train_data = preprocessor.transform_to_indices(train_data)

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_train_data, pkl_file)

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    dev_data = preprocessor.read_scifact(devset)

    print("\t* Transforming words in premises and hypotheses to indices...")
    transformed_dev_data = preprocessor.transform_to_indices(dev_data)
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(transformed_dev_data, pkl_file)

    # -------------------- Embeddings preprocessing -------------------- #
    print(20*"=", " Preprocessing embeddings ", 20*"=")
    print("\t* Building embedding matrix and saving it...")
    embed_matrix = preprocessor.build_embedding_matrix(embeddings_file)
    with open(os.path.join(targetdir, "embeddings.pkl"), "wb") as pkl_file:
        pickle.dump(embed_matrix, pkl_file)

    with open(os.path.join(targetdir, "preprocessor.pkl"), "wb") as pkl_file:
        pickle.dump(preprocessor, pkl_file)


if __name__ == "__main__":
    default_config = "../../config/scifact/scifact_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the scifact datasets")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing scifact"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"]
    )
