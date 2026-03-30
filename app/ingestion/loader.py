from beir import util
from beir.datasets.data_loader import GenericDataLoader

def load_fiqa():
    dataset_path = util.download_and_unzip(
        "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip",
        "datasets"
    )

    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    print(len(corpus))
    print(len(queries))
    print(len(qrels))


    return corpus, queries, qrels
    