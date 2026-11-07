SYSTEM = "You are a careful question-answering assistant. Answer only from given passages."

FOLLOWUP_PROMPT = """You are doing multi-step retrieval. A first search found these passages:

{passages}

Original question: {question}

The passages above are not enough to answer the question. Write ONE short search query (5-12 words) that targets the missing information. Output ONLY the query, no explanation."""

ANSWER_PROMPT = """Passages:
{passages}

Question: {question}

Answer in a few words, or yes/no. Do not explain."""

LLM_ONLY_PROMPT = """Question: {question}

Answer in a few words, or yes/no."""

def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(f"[{i+1}] {p['text']}" for i, p in enumerate(passages))