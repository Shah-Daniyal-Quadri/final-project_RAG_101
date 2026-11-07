# rag101-multihop

Multi-hop RAG over a curated Wikipedia collection. Compares three systems on
the same question set:

- **llm_only**: Llama 3.1 8B answers from its own knowledge.
- **rag**: single-shot retrieval of top-4 passages, then answer.
- **multihop**: retrieve top-2, ask the LLM for a follow-up query,
  retrieve 2 *new* passages, then answer from all 4.

## 0. Prerequisites

- Python 3.10+
- Windows / macOS / Linux
- ~5 GB free disk (Llama 3.1 8B + Python deps)

## 1. Set up Ollama

1. Install Ollama from https://ollama.com/download (one-click installer
   for Windows/macOS, or `curl -fsSL https://ollama.com/install.sh | sh`
   on Linux).
2. Open a **new terminal** and confirm it's on PATH:

   ```powershell
   ollama --version
   ```

3. Start the Ollama server (leave this terminal running):

   ```powershell
   ollama serve
   ```

   On Windows/macOS the desktop app already runs the server in the
   background, so `ollama serve` is only needed if you installed the CLI
   manually.

4. Pull the model:

   ```powershell
   ollama pull llama3.1:8b
   ```

   Confirm it's there:

   ```powershell
   ollama list
   ```

   You should see a line starting with `llama3.1:8b`. Copy that exact
   string — it needs to match `config.yaml` (see below).

5. Smoke test the model:

   ```powershell
   ollama run llama3.1:8b "Say OK"
   ```

   Type `/bye` to exit.

## 2. Set up the Python environment

From the repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate          # macOS / Linux

pip install -r requirements.txt
```

Expected packages: `langchain`, `langchain-ollama`, `langchain-huggingface`,
`sentence-transformers`, `numpy`, `pandas`, `pyyaml`, `requests`, `tqdm`.

## 3. Configure the model

Open `config.yaml` and confirm:

```yaml
model: llama3.1:8b
embedding_model: all-MiniLM-L6-v2
temperature: 0
seed: 42
top_k_first: 2
top_k_second: 2
top_k_single: 4
ollama_host: http://localhost:11434
```

`model:` must exactly match the name from `ollama list`. If `ollama list`
shows `llama3.1:8b-instruct-q4_K_M`, use that string instead.

## 4. Run the full pipeline (first time only)

Run these from the repo root, in order:

```powershell
# 4a. Fetch the 15 Wikipedia articles listed in src/collect_articles.py
python -m src.collect_articles
#    -> data/raw/articles.json

# 4b. Clean and chunk the article text
python -m src.preprocess
#    -> data/processed/chunks.jsonl  (one chunk per line)

# 4c. Embed the chunks and build the search index
python -m src.embed_index
#    -> data/processed/index.npz

# 4d. Write seed questions and freeze dev/test splits
python bootstrap_seed.py
#    -> data/processed/questions.jsonl
#    -> data/splits/dev.json, data/splits/test.json
```

Sanity-check what landed:

```powershell
dir data\raw\articles.json
dir data\processed\chunks.jsonl
dir data\processed\index.npz
dir data\processed\questions.jsonl
dir data\splits\dev.json
```

## 5. Run the experiments

Each command runs one system on the dev split and writes
`results/<system>_dev.csv` plus prints an aggregate JSON:

```powershell
python -m src.run_experiment --system llm_only  --split dev
python -m src.run_experiment --system rag       --split dev
python -m src.run_experiment --system multihop  --split dev
```

Sample output line per question:

```
[q006] 'Abbasid' | EM=1 F1=1.00
```

Final aggregate:

```json
{"system": "rag", "split": "dev", "n": 12, "EM": 0.5, "F1": 0.5,
 "Recall@4": 0.0, "avg_latency_s": 8.55}
```

Run on test split only **after** you've frozen the question set:

```powershell
python -m src.run_experiment --system multihop --split test
```

## 6. Reading the results

Each `results/<system>_dev.csv` has one row per question:

| Column | Meaning |
|---|---|
| `qid` | Question ID |
| `question` | Question text |
| `gold` | Gold answer |
| `pred` | System's predicted answer |
| `em` | Exact match (0/1) |
| `f1` | Token-level F1 (0–1) |
| `recall_at_4` | Fraction of `supporting_chunks` in top-4 retrieved (or `NaN`) |
| `retrieved` | `|`-separated chunk IDs retrieved |
| `followup_query` | (multihop only) query generated for hop 2 |
| `latency_s` | Wall-clock seconds for this question |

Aggregate metrics across the three systems become the Checkpoint 2 results
table.

## 7. Repo layout

```
rag101-multihop/
├── config.yaml
├── bootstrap_seed.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/articles.json
│   ├── processed/{chunks.jsonl, index.npz, questions.jsonl}
│   └── splits/{dev,test}.json
├── src/
│   ├── config.py
│   ├── collect_articles.py
│   ├── preprocess.py
│   ├── embed_index.py
│   ├── retriever.py
│   ├── llm.py
│   ├── prompts.py
│   ├── baselines.py
│   ├── multihop.py
│   ├── evaluate.py
│   └── run_experiment.py
├── results/
└── tests/test_smoke.py
```

## 8. Troubleshooting

| Error | Fix |
|---|---|
| `Could not find a version that satisfies the requirement requirements.txt` | You typed `pip install requirements.txt`. Use `pip install -r requirements.txt`. |
| `FileNotFoundError: data/splits/dev.json` | You skipped `python bootstrap_seed.py`. Run it before `run_experiment`. |
| `Connection refused` from ChatOllama | Ollama server not running. `ollama serve` in another terminal. |
| `model 'llama3.1:8b' not found` | Run `ollama pull llama3.1:8b`, then confirm `config.yaml` `model:` matches `ollama list`. |
| `Recall@4` shows `null` or `0.0` | The questions' `supporting_chunks` are empty. Fill them in (see Message 1). |
| Multihop scores worse than rag | Check `followup_query` column. Small models echo the prompt; use `llama3.1:8b`. |

## 9. Reproducibility

- `config.yaml` pins model, embedding, temperature (0), seed (42), top-k.
- `data/splits/{dev,test}.json` are frozen files, never regenerated per run.
- `results/*.csv` contain per-question predictions and retrieved chunk IDs.
- Commit `config.yaml` and `data/splits/` before reporting any numbers.