import torch
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from logger import log


class SemanticCache:
    """Stores query embeddings and finds matches using SentenceTransformer similarity."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", maxsize: int = 200):
        self.model = SentenceTransformer(model_name)
        self.maxsize = maxsize
        self.cache = []  # List of dicts: {"query": str, "response": str, "embedding": torch.Tensor}

    def get(self, query: str, threshold: float = 0.85):
        """Looks for a semantically similar query in cache."""
        if not self.cache:
            return None, 0.0

        query_emb = self.model.encode(query, convert_to_tensor=True)
        stored_embs = torch.stack([item["embedding"] for item in self.cache])

        # Compute cosine similarity across all stored queries
        similarities = self.model.similarity(query_emb, stored_embs)[0]
        best_idx = int(torch.argmax(similarities).item())
        best_score = float(similarities[best_idx].item())

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
        """Embeds query as tensor and stores it with FIFO eviction."""
        if len(self.cache) >= self.maxsize:
            self.cache.pop(0)

        embedding = self.model.encode(query, convert_to_tensor=True)
        self.cache.append(
            {
                "query": query,
                "response": response,
                "embedding": embedding,
            }
        )


class TwoLayerCache:
    """Combines an exact TTLCache with a SemanticCache."""

    def __init__(self, ttl: int = 3600, maxsize: int = 500, semantic_threshold: float = 0.85):
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
app_cache = TwoLayerCache(semantic_threshold=0.85)


if __name__ == "__main__":
    print("Testing TwoLayerCache in isolation...")

    # Seed the cache
    initial_q = "What is retrieval augmented generation?"
    initial_a = "Retrieval Augmented Generation (RAG) grounds an LLM with external document context."
    app_cache.set(initial_q, initial_a)

    # 1. Exact match test (with casing difference)
    res1, src1, score1 = app_cache.get("what is retrieval augmented generation?")
    print(f"Test 1 (Exact): Source = {src1} | Score = {score1}")

    # 2. Semantic match test (paraphrase)
    res2, src2, score2 = app_cache.get("Explain retrieval augmented generation to me.")
    print(f"Test 2 (Semantic): Source = {src2} | Score = {round(score2, 3)}")

    # 3. Cache miss test (unrelated question)
    res3, src3, score3 = app_cache.get("How do I bake sourdough bread?")
    print(f"Test 3 (Miss): Source = {src3} | Score = {round(score3, 3)}")