from langchain_core.tools import tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_exa import ExaSearchResults
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

exa = ExaSearchResults(max_results=10, api_key=os.getenv("EXA_API_KEY"))

def google_local_search(query: str, location: str, language: str, country: str) -> List[Dict[str, Any]]:
    """Search Google via SerpAPI with dynamic location and language context.
    Args:
        query: The search terms or question (e.g., "top historical sites in Kyoto").
        location: The target location city (e.g., "Kyoto, Japan" or "Berlin, Germany").
        language: The 2-letter language code of the language spoken locally in the destination (e.g., "en", "ja", "de").
        country: The 2-letter country code for the destination location (e.g., "de", "fr", "ca")"""
    serp = SerpAPIWrapper(serpapi_api_key= os.getenv("SERPAPI_API_KEY"), params = {
        "engine": "google",
        "google_domain": f"google.{country}",
        "hl": language,
        "gl": country,
        "location": location
    })

    search = serp.results(query=query)

    # Return organic results cleanly
    organic_results = search.get("organic_results", [])
    return [
        {
            "title": r.get("title"),
            "link": r.get("link"),
            "snippet": r.get("snippet")
        }
        for r in organic_results[:5] 
        ]


def exa_semantic_search(query: str) -> List[Dict[str,Any]]:
    """Use to discover local blogs, Reddit posts, and community discussions."""
    search = exa(query)
    results = search.get_dict()

    # Return organic results cleanly
    organic_results = results.get("organic_results", [])
    for i, r in enumerate(organic_results, 1):
        title = r.get("title", "Unknown"),
        link = r.get("link", ""),
        snippet = r.get("content", "").strip()
        if len(snippet) >300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."

        organic_results.append(f"{i}. **{title}**\n {link}\n {snippet}")
    return "\n\n".join(organic_results)

all_search_tools = [google_local_search, exa_semantic_search]