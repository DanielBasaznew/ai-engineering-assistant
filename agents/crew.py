import os
import re
import time
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# 1. Configure Gemini 3.1 Flash-Lite
llm = LLM(
    model="gemini/gemini-3.1-flash-lite",
    api_key=api_key,
    temperature=0.2,
)


# 2. Hardened Web Search Tool with Automatic Query Simplification
@tool("web_search")
def web_search(query: str) -> str:
    """Search DuckDuckGo for recent and reliable information on a given query."""

    def clean_query(q: str) -> str:
        # Strip quotes and special punctuation
        q_clean = re.sub(r'["\':\(\)\[\]]', " ", q)
        # Keep only the first 5 essential words for clean search matching
        words = [w for w in q_clean.split() if len(w) > 2]
        return " ".join(words[:5])

    try:
        # 1. First attempt with cleaned query
        cleaned = clean_query(query)
        results = DDGS().text(cleaned, max_results=4)

        # 2. Fallback attempt if zero results returned
        if not results:
            fallback_q = " ".join(query.split()[:3])
            results = DDGS().text(fallback_q, max_results=4)

        if not results:
            return f"No results found for query: {query}"

        formatted = []
        for r in results:
            title = r.get("title", "No Title")
            snippet = r.get("body", "No Snippet")
            formatted.append(f"- {title}: {snippet}")
        return "\n".join(formatted)
    except Exception as e:
        return f"Search failed with error: {str(e)}"


# 3. Define Hardened Agents with Iteration & Retry Limits
researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct deep research on emerging tech topics. Search at least twice with different queries to gather verified facts, statistics, and trends.",
    backstory=(
        "You are an elite research analyst specializing in technology trends. "
        "You always perform multiple distinct web searches to corroborate claims, "
        "synthesize real data points, and deliver thorough briefings. "
        "You never write consumer articles or editorial critiques."
    ),
    tools=[web_search],
    llm=llm,
    verbose=True,
    max_iter=5,
    max_retry_limit=2,
)

writer = Agent(
    role="Tech Content Strategist",
    goal="Transform technical briefings into executive-level, clear, and comprehensive reports for leadership.",
    backstory=(
        "You are a master technical writer and executive communications strategist. "
        "You translate raw bullet points and technical data into polished, authoritative prose "
        "with clear headings, executive summaries, and detailed analysis."
    ),
    llm=llm,
    verbose=True,
    max_iter=5,
    max_retry_limit=2,
)

reviewer = Agent(
    role="Editorial Quality Assurance Specialist",
    goal="Audit drafts against research briefings for factual accuracy, structural integrity, and word count minimums.",
    backstory=(
        "You are a meticulous technical editor. You review drafts line by line against source briefings, "
        "verify structural compliance, confirm depth and tone, and issue formal APPROVED or REVISION NEEDED verdicts."
    ),
    llm=llm,
    verbose=True,
    max_iter=5,
    max_retry_limit=2,
)


def run_research_crew(topic: str) -> dict:
    """Runs the hardened 3-agent research team and saves the result to a markdown file."""
    print(f"\n==================================================")
    print(f"  Starting Research Run: {topic}")
    print(f"==================================================\n")

    start_time = time.time()

    # 4. Define Modular Tasks with Strict Quality Contracts
    research_task = Task(
        description=(
            f"Conduct comprehensive research on: '{topic}'.\n"
            "Requirements:\n"
            "1. Search at least twice using different, targeted search queries.\n"
            "2. Identify at least 3 concrete trends, backed by specific industry facts, statistics, or analyst predictions.\n"
            "3. Provide a structured research briefing of at least 300 words."
        ),
        expected_output="A comprehensive research briefing with at least 3 key trends, facts/statistics, and a minimum of 300 words.",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Using the research briefing provided, author a leadership-ready executive report.\n"
            "Requirements:\n"
            "1. Use the following Markdown structure:\n"
            "   ## [Title]\n"
            "   **Executive Summary**\n"
            "   **Key Industry Findings** (addressing each trend with details)\n"
            "   **Strategic Implications for Leadership**\n"
            "2. Ensure the report is comprehensive, authoritative, and contains a minimum of 400 words."
        ),
        expected_output="A polished, multi-section executive report adhering to the requested headers and clearing at least 400 words.",
        agent=writer,
    )

    review_task = Task(
        description=(
            "Audit the writer's report against the original research briefing.\n"
            "Requirements:\n"
            "1. Verify factual accuracy against the briefing.\n"
            "2. Confirm adherence to required sections and minimum 400-word count.\n"
            "3. Conclude with an explicit verdict header: 'VERDICT: APPROVED' or 'VERDICT: REVISION NEEDED' followed by a brief critique."
        ),
        expected_output="An editorial QA audit noting fidelity, length, and structure, concluding with VERDICT: APPROVED or VERDICT: REVISION NEEDED.",
        agent=reviewer,
    )

    # 5. Assemble Sequential Crew
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
    )

    result = crew.kickoff()
    elapsed_time = round(time.time() - start_time, 2)
    output_text = str(result)
    word_count = len(output_text.split())

    # 6. Save Artifact to Disk with Safe Slug Handling
    os.makedirs("reports", exist_ok=True)
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", topic.lower()).strip("_")
    # Truncate and guard against empty slugs (e.g. if topic was purely punctuation)
    slug = slug[:40] if slug else "research_report"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join("reports", f"{slug}_{timestamp}.md")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# Research Report: {topic}\n\n")
        f.write(f"**Date Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Execution Time:** {elapsed_time} seconds\n")
        f.write(f"**Total Words:** {word_count}\n\n")
        f.write("---\n\n")
        f.write(output_text)

    print(f"\n--------------------------------------------------")
    print(f"  Run Completed in {elapsed_time}s | Words: {word_count}")
    print(f"  Saved report to: {filepath}")
    print(f"--------------------------------------------------\n")

    return {
        "topic": topic,
        "elapsed_time": elapsed_time,
        "word_count": word_count,
        "filepath": filepath,
        "output": output_text,
    }


if __name__ == "__main__":
    import sys

    print("\n--- Production Multi-Agent Research System ---")
    if len(sys.argv) > 1:
        user_topic = " ".join(sys.argv[1:]).strip()
    else:
        user_topic = input("Enter a research topic (or press Enter for default): ").strip()

    if not user_topic:
        user_topic = "Open Source LLMs vs Proprietary Models Enterprise Adoption Trends 2025 2026"
        print(f"No topic entered. Using default: '{user_topic}'")

    run_research_crew(user_topic)
