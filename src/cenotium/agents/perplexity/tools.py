"""Perplexity search tool for web queries."""

import os

import requests
from langchain.tools import Tool


def perplexity_search(query: str) -> str:
    """Search the internet using Perplexity API."""
    api_url = "https://api.perplexity.ai/chat/completions"
    api_key = os.getenv("PERPLEXITY_KEY")

    if not api_key:
        return "Error: PERPLEXITY_KEY environment variable not set"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "sonar-pro",
        "messages": [{"role": "user", "content": query}],
    }

    response = requests.post(api_url, headers=headers, json=payload, timeout=30)

    if response.status_code == 200:
        result = response.json()
        return result["choices"][0]["message"]["content"]
    return f"Error: {response.status_code} - {response.text}"


perplexity_tool = Tool(
    name="PerplexitySearch",
    func=perplexity_search,
    description="Search the internet via the Perplexity API. Input: search query string.",
)
