import os
import time
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone


api_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def add_items(index_name: str, data_csv: str):
    df = pd.read_csv(data_csv)
    print(df.head())
    print(df.columns)
    print(df.dtypes)

    plots = df["Plot"].dropna().tolist()
    titles = df["Title"].dropna().tolist()

    batch_size = 100
    for i in range(0, len(plots), batch_size):
        batch_plots = plots[i:i+batch_size]
        batch_titles = titles[i:i+batch_size]
        embeddings = embedding_model.encode(batch_plots).tolist()
        ids = [f"plot-{i+j}" for j in range(len(batch_plots))]
        metadata = [{"title": title} for title in batch_titles]

        pc.upsert(
            index_name=index_name,
            vectors=zip(ids, embeddings, metadata)
        )


def search(index_name: str, query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query]).tolist()[0]
    results = pc.query(
        index_name=index_name,
        queries=[query_embedding],
        top_k=top_k,
        include_metadata=True
    )
    return results['results'][0]['matches']


def main():
    index_name = "movie-plots1"

    if not pc.has_index(index_name):
        pc.create_index(
                name=index_name,
                dimension=384,
                metric="cosine",
                spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
            )

    # Time how long it takes to add items
    start_time = time.time()
    add_items(index_name, "wiki_movie_plots_first_100.csv")
    end_time = time.time()
    print(f"\nIndexing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
