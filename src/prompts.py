"""
Prompt templates for all three systems.

Design notes
------------
- No yes/no wording anywhere: the dataset has no yes/no questions, and
  mentioning it made the model answer "yes"/"no" to normal questions.
- The model always gives its best answer (never "unknown"). If the passages
  don't help, it falls back on its own knowledge.
- All three systems share the same answer rules (ANSWER_RULES); the only
  difference is whether passages are included. This keeps the comparison fair.
- Prompts end with "Answer:" so the model replies with the answer itself
  (prevents replies like "I'm ready to answer.").
- Keep answers SHORT: EM/F1 compare against gold strings like "al-Ma'mun" or
  "1572", so anything more than a few words hurts scoring.
- The follow-up prompt gives the model concrete criteria for what makes a good
  search query, rather than a vague "write a short query".
"""

_SYSTEM_BASE = (
    "You are a careful factual question-answering assistant. "
    "Always give your best answer, even if you are unsure. "
    "Never reply \"unknown\" and never refuse. "
    "Reply with only the answer itself, never a full sentence or an explanation."
)

# RAG + multihop: passages are provided, but the model may fall back on its own knowledge.
SYSTEM = (
    _SYSTEM_BASE + " "
    "Use the provided passages when they are helpful; if they do not contain "
    "the answer, answer from your own knowledge."
)

# LLM-only: no passages are provided.
SYSTEM_LLM_ONLY = _SYSTEM_BASE + " Answer from your own knowledge."

ANSWER_RULES = """\
Rules:
  - Reply with only the answer: a name, place, title, term, or year.
  - For "who/which ... first" or "who lived longer" questions, reply with the name of that person or event.
  - Give your best answer even if you are unsure.
  - Do not write a sentence. Do not explain."""

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
Passages:
{passages}

Question: {question}

Use the passages if they help; otherwise answer from your own knowledge.
""" + ANSWER_RULES + """

Answer:"""

LLM_ONLY_PROMPT = """\
Question: {question}

""" + ANSWER_RULES + """

Answer:"""


def format_passages(passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages):
        title = p.get("title", "unknown")
        section = p.get("section", "")
        subsection = p.get("subsection", "")
        if section:
            label = f"{title} > {section}"
            if subsection:
                label += f" > {subsection}"
        else:
            label = title
        parts.append(f"[{i + 1}] ({label}) {p['text']}")
    return "\n\n".join(parts)