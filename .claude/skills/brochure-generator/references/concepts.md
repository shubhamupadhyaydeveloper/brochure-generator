# Deep Concepts — Brochure Generator Project

## Tokens — Why They Matter

A token ≈ 0.75 words. "Hello world" = 2 tokens. "Tokenization" = 1 token.

Why you care:
- Every LLM has a **context window** = max tokens it can process at once
- GPT-4o-mini: 128k tokens | Claude Haiku: 200k tokens
- You pay per token (input + output)
- Longer input = slower response + higher cost

That's why `text[:5000]` truncation in the scraper is intentional — most company
homepages don't need more than 5000 chars to extract key brochure info.

---

## Streaming — How It Actually Works

Under the hood, streaming uses **SSE (Server-Sent Events)** — the server keeps
the HTTP connection open and pushes small chunks of data as they're generated.

Without streaming:
```
Model generates entire response (10-15 sec) → sends everything at once
```

With streaming:
```
Model generates token → sends it immediately → generates next → sends → ...
```

In Python, `stream=True` returns a generator object. Each iteration gives you
one chunk. `yield` in your function makes YOUR function a generator too —
which is what Gradio's streaming output expects.

---

## System Prompt vs User Prompt

| | System Prompt | User Prompt |
|---|---|---|
| Role | Sets the model's persona and rules | The actual task/question |
| Persistence | Stays constant across turns | Changes each turn |
| Priority | Higher weight in model's attention | Lower weight |
| Use for | Tone, format rules, constraints | The content to process |

For brochure generator:
- System: "You are a copywriter, output markdown, use these sections..."
- User: "Here is the raw website content: ..."

---

## Temperature

Controls randomness. Range: 0.0 to 2.0 (usually 0 to 1).

- `0.0` → Deterministic, same output every time (good for code, data extraction)
- `0.7` → Balanced creativity (good for brochures, marketing copy)
- `1.0+` → Very creative/unpredictable (good for brainstorming, poetry)

For this project use `temperature=0.7` — you want creative writing but not chaos.

---

## Why WebBaseLoader (LangChain) vs raw requests?

`requests + BeautifulSoup` = manual, more control, no extra dependencies

`WebBaseLoader` (LangChain) = convenience wrapper, handles some edge cases,
but adds LangChain as a dependency (heavy, opinionated framework).

Ed Donner's course uses both — understanding the manual approach first
(requests/BS4) makes you appreciate what LangChain abstracts away.

---

## Multi-Model Architecture Pattern

The brochure generator teaches a key architectural pattern:
write model-agnostic business logic, swap the LLM client underneath.

```
scrape_website()     ← model-agnostic
build_messages()     ← model-agnostic  
call_llm(model)      ← model-specific (OpenAI vs Anthropic API differ here)
display_output()     ← model-agnostic
```

This pattern matters for production — you never want your app to be
100% locked into one provider.
