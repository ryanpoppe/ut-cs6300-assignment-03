
from __future__ import annotations
import shutil
from pathlib import Path
import pprint
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pandas import read_csv
import time
import json


# --------- Config ----------
DB_DIR = Path("./chroma_demo_db")
COLLECTION_NAME = "vector_db_demo"
DATASET_PATH = "wiki_movie_plots_deduped.csv"
EMBEDDER = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")
pp = pprint.PrettyPrinter(indent=2, width=100)


# --------- Helpers ----------
def fresh_start():
    """Delete the demo DB so runs are reproducible in class."""
    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)


def get_client():
    """
    Use the modern PersistentClient if available (Chroma >= 0.5),
    otherwise fall back to the older Settings-based client.
    """
    try:
        # Chroma >= 0.5
        return chromadb.PersistentClient(path=str(DB_DIR))
    except AttributeError:
        # Older Chroma
        from chromadb.config import Settings
        return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(DB_DIR)))


def show(title: str, obj):
    print(f"\n=== {title} ===")
    pp.pprint(obj)


def add_items(collection: chromadb.api.models.Collection.Collection, csv_file: str, max_batch_size: int):
    """
    Create a Chroma collection from a CSV file containing movie plots.
    The CSV has the following columns:
    Release Year,Title,Origin/Ethnicity,Director,Cast,Genre,Wiki Page,Plot

    Args:
        collection: Chroma collection to add items to
        csv_file: Path to CSV file
    """
    df = read_csv(csv_file)
    print(df.head())
    print(df.columns)
    print(df.dtypes)
    print(df.shape)
    print(df.info())
    print(df.describe())
    print(df['Title'].head())
    print(df['Genre'].value_counts())

    ids = df.index.astype(str).tolist()
    metadatas = df.drop(columns=['Plot']).to_dict(orient='records')
    documents = df['Plot'].tolist()
    for i in range(0, len(documents), max_batch_size):
        batch_ids = ids[i:i + max_batch_size]
        batch_metadatas = metadatas[i:i + max_batch_size]
        batch_documents = documents[i:i + max_batch_size]
        collection.add(ids=batch_ids, metadatas=batch_metadatas, documents=batch_documents)
        print(f"Added batch {i // max_batch_size + 1} with {len(batch_ids)} items")


def search(collection: chromadb.api.models.Collection.Collection, query: str, n_results=3):
    return collection.query(query_texts=[query], n_results=n_results)


def main():
    fresh_start()
    client = get_client()
    print(f"ChromaDB max batch size: {client.get_max_batch_size()}")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDER,
        metadata={"hnsw:space": "cosine"}
    )

    # Time how long it takes to add items
    start_time = time.time()
    add_items(collection, DATASET_PATH, client.get_max_batch_size())
    end_time = time.time()
    print(f"\nIndexing time: {end_time - start_time:.2f} seconds")
    disk_uage_size = sum(f.stat().st_size for f in DB_DIR.rglob('*') if f.is_file())
    print(f"Disk usage: {disk_uage_size / 1024:.2f} KB")

    # load queires from queries.json
    queries = []
    with open('queries.json', 'r') as f:
        queries = json.load(f)
    print(f"\nLoaded {len(queries)} queries from queries.json")

    # Keep track of Query latency (P50, P95, P99) (this is across your full set of queries)
    # Also track query results and save to results.json
    # Format: { "query": "...", "results": [...] }
    results = []
    latencies = []
    for query in queries:
        start_time = time.time()
        search_results = search(collection, query, n_results=5)
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        results.append({"query": query, "latency": latency, "results": search_results})

    # Save results to results.json
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to results.json")

    latencies.sort()
    n = len(latencies)
    p50 = latencies[int(0.5 * n)]
    p95 = latencies[int(0.95 * n)]
    p99 = latencies[int(0.99 * n)]
    print(f"\nQuery latencies (seconds): P50={p50:.2f}, P95={p95:.2f}, P99={p99:.2f}")


if __name__ == "__main__":
    main()
