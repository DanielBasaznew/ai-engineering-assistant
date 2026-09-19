import numpy as np
from cachetools import TTLCache
from logger import log
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:
    """Stores query embeddings and finds matches using cosine similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", maxsize: int = 200):
        # Load small, efficient local embedding model
        self.model = SentenceTransformer(model_name)
        self.maxsize = maxsize
        self.cache = []  # List of dicts: {"query": str, "response": str, "embedding": np.ndarray}

    def get(self, query: str, threshold: float = 0.88):
        """Looks for a semantically similar query in cache."""
        if not self.cache:
            return None, 0.0

        query_emb = self.model.encode([query])
        stored_embs = np.array([item["embedding"] for item in self.cache])

        # Compute cosine similarity across all stored queries in one vectorized step
        similarities = cosine_similarity(query_emb, stored_embs)[0]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            matched_query = self.cache[best_idx]["query"]
            log.info(
                "Semantic cache hit",
                extra={
                    "similarity": round(best_score, 4),
                    "matched_query": matched_query,
                    "incoming_query": query,
                },
            )
            return self.cache[best_idx]["response"], best_score

        return None, best_score

    def set(self, query: str, response: str):
        """Embeds query and stores the record with FIFO eviction if maxsize is reached."""
        if len(self.cache) >= self.maxsize:
            self.cache.pop(0)  # Evict oldest entry

        embedding = self.model.encode(query)
        self.cache.append(
            {
                "query": query,
                "response": response,
                "embedding": embedding,
            }
        )


class TwoLayerCache:
    """Combines an exact TTLCache with a SemanticCache."""

    def __init__(self, ttl: int = 3600, maxsize: int = 500, semantic_threshold: float = 0.88):
        self.exact_cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.semantic_cache = SemanticCache(maxsize=maxsize)
        self.semantic_threshold = semantic_threshold

    def get(self, query: str):
        """Checks exact cache first, then semantic cache."""
        normalized_key = query.strip().lower()

        # 1. Exact match
        if normalized_key in self.exact_cache:
            log.info("Exact cache hit", extra={"query": query})
            return self.exact_cache[normalized_key], "exact_cache", 1.0

        # 2. Semantic match
        semantic_response, similarity = self.semantic_cache.get(
            query, threshold=self.semantic_threshold
        )
        if semantic_response:
            return semantic_response, "semantic_cache", similarity

        return None, "miss", similarity

    def set(self, query: str, response: str):
        """Populates both cache layers."""
        normalized_key = query.strip().lower()
        self.exact_cache[normalized_key] = response
        self.semantic_cache.set(query, response)


# Global instance
app_cache = TwoLayerCache()


if __name__ == "__main__":
    print("Testing TwoLayerCache in isolation...")

    # Seed an entry
    initial_q = "What is Retrieval Augmented Generation?"
    initial_a = "RAG combines an information retrieval system with a generative LLM."
    app_cache.set(initial_q, initial_a)

    # 1. Test exact repeat (with case change)
    res, src, score = app_cache.get("what is retrieval augmented generation?")
    print(f"Test 1 (Exact): Source = {src} | Score = {score}")

    # 2. Test semantic rephrasing
    res, src, score = app_cache.get("Can you explain what RAG means?")
    print(f"Test 2 (Semantic): Source = {src} | Score = {round(score, 3)}")

    # 3. Test completely unrelated query
    res, src, score = app_cache.get("How do I bake sourdough bread?")
    print(f"Test 3 (Miss): Source = {src} | Score = {round(score, 3)}")