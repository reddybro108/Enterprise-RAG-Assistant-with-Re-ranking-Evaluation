from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    embeddings = model.encode(texts)
    print("Embedding shape:", len(embeddings), len(embeddings[0]))
    return embeddings


# test run
if __name__ == "__main__":
    sample = ["This is a test sentence"]
    embed_texts(sample)