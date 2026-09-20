import os
import chromadb
from chromadb.utils import embedding_functions

# 1. Define where the persistent database will live on your hard drive
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")

# 2. Set up the persistent ChromaDB client
client = chromadb.PersistentClient(path=DB_PATH)

# 3. Configure the automatic embedding function using all-MiniLM-L6-v2
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection(collection_name: str = "knowledge_base"):
    """
    Retrieves an existing collection or creates a new one.
    Configured to use Cosine Similarity for vector comparisons.
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def distance_to_similarity(distance: float) -> float:
    """
    Converts ChromaDB cosine distance (0=identical, 1=unrelated)
    into an intuitive similarity score (1=identical, 0=unrelated).
    """
    return 1.0 - distance


def add_documents(collection, documents: list, metadatas: list, ids: list):
    """
    Adds documents, metadata dictionaries, and unique string IDs to a collection.
    Using .upsert() instead of .add() prevents crash errors if an ID already exists!
    """
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    print(f"[SUCCESS] Upserted {len(documents)} documents into '{collection.name}'.")


def search(collection, query: str, top_k: int = 3, filter_metadata: dict = None):
    """
    Automatically embeds the query string and returns top_k similar documents.
    Formats raw ChromaDB outputs into clean dictionaries with intuitive similarity scores.
    """
    # Build query arguments
    query_args = {"query_texts": [query], "n_results": top_k}
    # Optional metadata filter
    if filter_metadata:
        query_args["where"] = filter_metadata

    raw_results = collection.query(**query_args)

    # Clean and format the raw ChromaDB output
    formatted_results = []
    # raw_results returns nested lists (one list per query), so we index [0]
    for i in range(len(raw_results["ids"][0])):
        dist = raw_results["distances"][0][i]
        formatted_results.append(
            {
                "id": raw_results["ids"][0][i],
                "document": raw_results["documents"][0][i],
                "metadata": raw_results["metadatas"][0][i],
                "distance": dist,
                "similarity": distance_to_similarity(dist),
            }
        )
    return formatted_results
