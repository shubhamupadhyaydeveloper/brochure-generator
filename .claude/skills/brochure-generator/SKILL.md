---
name: brochure-generator
description: Helps build an AI-powered company brochure generator project from Ed Donner's LLM Engineering course. Use when user mentions "brochure generator", "brochure project", "company brochure", "scrape website brochure", "Week 3", "Week 4", or asks about streaming LLM output, WebBaseLoader, website scraping for LLM input, or multi-model brochure generation. Also use when debugging issues with OpenAI/Anthropic streaming, streamlit UI, or LangChain document loaders in a Jupyter notebook context.
---

# Brochure Generator — LLM Course Project Guide

This is an AI-powered tool that takes a company's website URL, scrapes its content,
and uses an LLM to generate a polished marketing brochure. Core learning goals:
multi-model usage, streaming output, prompt engineering, and basic UI with streamlit.

---

## Project Architecture

```
User enters URL
      ↓
Scrape website content (WebBaseLoader or requests + BeautifulSoup)
      ↓
Clean & truncate text (fit within context window)
      ↓
Send to LLM with a well-crafted system prompt
      ↓
Stream the response back to the UI (Streamlit)
      ↓
Display formatted brochure
```

---

## Step-by-Step Build Guide

### Step 1: Setup & Imports
```python
import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI  # or anthropic
import streamlit as st

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Common mistake**: Forgetting `load_dotenv()` before reading env vars.
Always call it before `os.getenv()`.

---

### Step 2: Website Scraper
```python
def scrape_website(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return text[:5000]  # Truncate — LLMs have context limits
```

**Why truncate?** GPT-4 has ~128k context but you're paying per token.
5000 chars ≈ ~1200 tokens — enough for most homepages.

**Why custom User-Agent?** Many sites block Python's default requests agent.

---

### Step 3: The System Prompt (Most Important Part)
```python
SYSTEM_PROMPT = """
You are a professional copywriter creating marketing brochures.
Given raw website content, extract key information and write a
compelling 1-page company brochure with these sections:

1. **Company Overview** — What they do, who they serve
2. **Key Products/Services** — Top 3-4 offerings
3. **Why Choose Them** — Unique value proposition
4. **Call to Action** — Contact info or next step

Write in a professional, engaging tone. Use markdown formatting.
If information is missing, write something reasonable based on context.
"""
```

**Concept — Why this prompt structure works:**
- Role assignment ("You are a copywriter") anchors the model's persona
- Explicit sections prevent the model from rambling
- "If missing, infer" prevents the model from refusing on sparse data

---

### Step 4: Streaming LLM Call
```python
def generate_brochure_stream(url: str):
    content = scrape_website(url)

    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Website content:\n\n{content}"}
        ],
        stream=True  # ← This is the key flag
    )

    result = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            result += delta
            yield result  # ← streamlit needs yield, not return
```

**Concept — Streaming vs non-streaming:**
- Without streaming: user waits 10-15 seconds, then sees full response
- With streaming: tokens appear word-by-word (like ChatGPT)
- `yield` instead of `return` makes this a Python generator — streamlit reads it progressively

---

### Step 5: Multi-Model Version (Course Extension)
```python
def generate_brochure(url: str, model_choice: str):
    content = scrape_website(url)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Website content:\n\n{content}"}
    ]

    if model_choice == "GPT-4o-mini":
        stream = openai_client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta: yield delta

    elif model_choice == "Claude Haiku":
        with anthropic_client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]
        ) as stream:
            for text in stream.text_stream:
                yield text
```

**Key difference — Anthropic vs OpenAI streaming API:**
- OpenAI: `chunk.choices[0].delta.content`
- Anthropic: uses `.stream()` context manager + `.text_stream` iterator

---

## Common Errors & Fixes

### `RateLimitError` or `AuthenticationError`
```
Check: print(os.getenv("OPENAI_API_KEY"))  # Should not be None
Fix: Make sure .env file is in same folder as your notebook
Fix: Make sure load_dotenv() is called before os.getenv()
```

### Scraper returns empty string
```
Cause: Site uses JavaScript rendering (React/Next.js SPA)
Fix: Use Selenium or Playwright instead of requests
Quick workaround: Try fetching /about or /company subpages
```

### streamlit shows all at once, not word-by-word
```
Cause: Using return instead of yield in your function
Fix: Change return result → yield result inside the for loop
```

### `timeout` error on scraping
```python
# Add timeout + error handling
try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
except Exception as e:
    return f"Could not scrape website: {e}"
```

### Brochure output is too generic
```
Fix: Make your system prompt more specific
Add: "Focus on technical differentiation, not generic marketing speak"
Add: "Here is the company name: {company_name}" as extra context
```

---

## Concepts to Understand Deeply

See `references/concepts.md` for detailed explanations of:
- What tokens are and why truncation matters
- How streaming works under the hood (SSE — Server Sent Events)
- Why system prompts vs user prompts behave differently
- Context window limits by model
- Temperature and when to change it (for brochures: use 0.7)

---

## Extending the Project (Portfolio Ideas)

1. **PDF export** — Use `fpdf2` or `reportlab` to save brochure as PDF
2. **Multiple pages** — Scrape `/about`, `/products`, `/contact` and merge
3. **Brand styling** — Ask user for brand color + tone, inject into prompt
4. **Comparison mode** — Generate brochures for 2 companies side by side
5. **BillSnap integration** — Generate a brochure for BillSnap itself using this tool
