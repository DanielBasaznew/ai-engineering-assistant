"""
Web Search Tool for AI Engineering Assistant (Week 4)
Formats DuckDuckGo search results into structured, truncated snippets.
"""

try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> str:
    """
    Searches DuckDuckGo for the given query and returns a formatted string
    of numbered search results with truncated snippets (max 200 chars).
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    try:
        results = []
        with DDGS() as ddgs:
            ddg_results = list(ddgs.text(query.strip(), max_results=max_results))

        if not ddg_results:
            return f"No results found for query: '{query}'"

        for idx, item in enumerate(ddg_results, start=1):
            title = item.get("title", "No Title")
            url = item.get("href", item.get("link", "No URL"))
            snippet = item.get("body", item.get("snippet", ""))

            # Truncate snippet to 200 characters
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."

            results.append(f"[{idx}] {title}\n    URL: {url}\n    Snippet: {snippet}")

        return "\n\n".join(results)

    except Exception as e:
        return f"Error executing web search: {str(e)}"


if __name__ == "__main__":
    print(web_search("Python news"))
