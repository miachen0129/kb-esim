import jsonlines

corpus_path = "datasets/meta/sciFact/corpus.jsonl"
corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus_path)}

def evidence_print(doc_id, inxs):
    abstract = corpus[doc_id]['abstract']
    evidence = " ".join(abstract[inx] for inx in inxs)
    print(evidence)
    return evidence

evidence_print(23557241, [3,4])