"""
Web Search Tool for AI Engineering Assistant (Week 4)
Formats DuckDuckGo search results into structured, truncated snippets.
"""

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Searches DuckDuckGo for the given query and returns a formatted string
    of numbered search results with truncated snippets (max 200 chars).
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    clean_query = query.strip()

    try:
        results = []
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(clean_query, max_results=max_results))

        if not ddg_results:
            return (
                f"No relevant web search results found for query: '{clean_query}'. "
                f"The search returned empty content. Suggest refining the search query or "
                f"synthesizing an answer from verified foundational knowledge."
            )

        total_snippet_chars = 0
        for idx, item in enumerate(ddg_results, start=1):
            title = item.get("title", "No Title")
            url = item.get("href", item.get("link", "No URL"))
            snippet = item.get("body", item.get("snippet", ""))

            # Truncate snippet to 200 characters
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            total_snippet_chars += len(snippet.strip())
            results.append(f"[{idx}] {title}\n    URL: {url}\n    Snippet: {snippet}")

        # Guard against weak/empty text across all returned entries
        if total_snippet_chars < 20:
            return (
                f"Web search for '{clean_query}' returned weak or insufficient snippet data "
                f"(under 20 characters). Suggest formulating a response based on existing context."
            )

        return "\n\n".join(results)

    except Exception as e:
        return (
            f"Web search attempt failed for query '{clean_query}': {str(e)}. "
            f"Search engine could not retrieve results."
        )


if __name__ == "__main__":
    print(web_search("Python news"))
