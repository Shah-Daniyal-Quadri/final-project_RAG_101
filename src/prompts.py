"""
Prompt templates for all three systems.

Design notes
------------
- The answer prompt branches on question shape (yes/no, who, when, where, which,
  or comparison) so the model is told the expected output format.
- The follow-up prompt gives the model concrete criteria for what makes a good
  search query, rather than a vague "write a short query".
- Keep answers SHORT: EM/F1 compare against gold strings like "al-Ma'mun" or
  "1572", so anything more than a few words hurts scoring.
"""

SYSTEM = (
    "You are a careful factual question-answering assistant. "
    "Answer only from the provided passages. "
    "If the passages do not contain the answer, respond with \"unknown\". "
    "Never explain your reasoning unless asked. "
    "Keep answers short: a few words, a number, a year, or yes/no."
)

FOLLOWUP_PROMPT = """\
You are on the second step of a two-step retrieval.

Passages retrieved so far:
{passages}

Original question: {question}

The passages above do NOT contain enough information to answer the question.

Write ONE short search query (5-12 words) that will retrieve the MISSING
information on a second Wikipedia search. Guidelines:
  - Name the specific entity, person, or concept that is missing.
  - Use keywords likely to appear in a Wikipedia article title or lead sentence.
  - Do NOT repeat the original question verbatim.
  - Do NOT explain your reasoning.
  - Output ONLY the query, on a single line, no quotes, no prefixes.

Query:"""

ANSWER_PROMPT = """\
Answer the question using ONLY the passages below.

Passages:
{passages}

Question: {question}

Format rules — follow the one that matches the question:
  - If the question starts with "Did", "Was", "Is", or "Are": answer "yes" or "no".
  - If the question starts with "Who": answer with the person's name only.
  - If the question starts with "When" or asks "what year": answer with the year only.
  - If the question starts with "Where": answer with the place name only.
  - If the question starts with "Which" or "What": answer with the specific entity, title, or thing.
  - If the question asks "who was born first / who died first / which came first":
    answer with the name of the earlier person or the earlier event.
  - Keep the answer to a few words. Do not write a full sentence. Do not explain."""

LLM_ONLY_PROMPT = """\
Question: {question}

Answer in a few words — a name, a year, a place, or yes/no. Do not explain."""


def format_passages(passages: list[dict]) -> str:
    """Render retrieved chunks with numbered markers and source titles."""
    return "\n\n".join(
        f"[{i + 1}] ({p.get('title', 'unknown')}) {p['text']}"
        for i, p in enumerate(passages)
    )