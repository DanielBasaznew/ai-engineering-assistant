import re
import torch
from cachetools import TTLCache
from sentence_transformers import SentenceTransformer
from logger import log


INELIGIBLE_SEMANTIC_PATTERNS = [
    # 1. Web search / fresh / dynamic
    r"(?i)\b(search(\s+the)?\s+web|web_search|browse(\s+the)?\s+web|look\s+up\s+online|google|duckduckgo)\b",
    r"(?i)\b(latest|recent|current|breaking|today|now|this\s+year|this\s+month|2026)\b",
    # 2. Python / code execution & MCP
    r"(?i)\b(use\s+python|run\s+python|execute\s+python|execute\s+code|run\s+code|code_executor|execute_python)\b",
    r"(?i)\b(calculate|compute)\s+.*(with|using|in)\s+python\b",
    r"(?i)\buse\s+code_executor\b",
    r"(?i)\b(mcp|mcp_explain_repository|explain_repository|repository_explainer|explain\s+repo|explain\s+folder|inspect\s+folder|use\s+mcp)\b",
    # 3. Document / PDF / RAG retrieval
    r"(?i)\b(my\s+uploaded\s+document|the\s+uploaded\s+document|uploaded\s+document|the\s+loaded\s+document|my\s+document|the\s+document|the\s+pdf|uploaded\s+pdf)\b",
    r"(?i)\b(read_pdf|read_pdf_page|search_knowledge_base)\b",
    r"(?i)\b(in\s+my\s+document|from\s+my\s+document|in\s+the\s+document|from\s+the\s+document|in\s+the\s+pdf|from\s+the\s+pdf)\b",
    r"(?i)\bcompare\s+.*with\s+.*(document|pdf)\b",
    # 4. Memory operations
    r"(?i)\b(remember\s+that|please\s+remember|my\s+preferred|what\s+is\s+my\s+(favorite|preferred|name|role))\b",
    # 5. Multi-step workflows
    r"(?i)\b1[\.\)]\s+.*2[\.\)]\s+",
    r"(?i)\bthen\s+compare\b",
]


def is_semantic_cache_eligible(query: str) -> bool:
    """
    Evaluates whether a query is safe for semantic cache matching.
    Returns False for queries requiring live web data, tool calls (Python/PDF/RAG),
    memory mutations/lookups, or multi-step continuation workflows.
    """
    clean_q = query.strip()
    for pattern in INELIGIBLE_SEMANTIC_PATTERNS:
        if re.search(pattern, clean_q):
            return False
    return True


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
        """Checks exact cache for safe deterministic requests, and semantic cache if eligible."""
        if not is_semantic_cache_eligible(query):
            log.info("Query ineligible for cache", extra={"query": query[:60]})
            return None, "miss", 0.0

        normalized_key = query.strip().lower()

        # 1. Exact match (for safe deterministic requests)
        if normalized_key in self.exact_cache:
            log.info("Exact cache hit", extra={"query": query})
            return self.exact_cache[normalized_key], "exact_cache", 1.0

        # 2. Semantic match
        semantic_response, similarity = self.semantic_cache.get(
            query, threshold=self.semantic_threshold
        )
        if semantic_response is not None:
            log.info("Semantic cache hit", extra={"similarity": similarity, "query": query})
            return semantic_response, "semantic_cache", similarity

        return None, "miss", similarity

    def set(self, query: str, response: str):
        """Populates exact and semantic cache only for safe deterministic queries."""
        if not is_semantic_cache_eligible(query):
            return
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