# KnowBase — Enterprise Knowledge-Base RAG

> Turn your internal docs into an always-on expert. KnowBase indexes policies, runbooks, and FAQs with hybrid retrieval (BM25 + TF-IDF cosine fused by reciprocal-rank fusion), answers questions with citation-grounded responses, evaluates retrieval quality, and operates fully offline.

## What KnowBase Does

- **Hybrid retrieval** — BM25 keyword + TF-IDF cosine fused via reciprocal-rank fusion
- **Citation-grounded answers** — every response links to the exact source document and passage
- **Dual answer modes** — Claude-generated with API key; extractive synthesis otherwise (always works)
- **Retrieval evaluation harness** — hit@1, hit@3, MRR measured on a QA test set
- **Document freshness** — incremental index updates without full rebuild

## Architecture

```
Internal Documents (PDF, Markdown, text)
    └─> DocumentIngester    (parse, chunk, clean)
    └─> BM25Index           (keyword retrieval)
    └─> TFIDFIndex          (cosine similarity retrieval)
    └─> RRFRanker           (reciprocal-rank fusion)
    └─> AnswerSynthesizer   (Claude API or extractive fallback)
    └─> EvalHarness         (hit@k, MRR measurement)
```

## Quickstart

```bash
python -m knowbase index --docs ./docs/
python -m knowbase ask "What is the PTO policy?"
python -m knowbase demo
```

## Test

```bash
python tests/test_smoke.py
```

---

## 👤 Author & Contact

- **Author**: Nathaniel Gordon
- **Role**: Senior AI & Machine Learning Engineer
- **GitHub**: [github.com/nathaniel-gordon](https://github.com/nathaniel-gordon)
- **Portfolio / Upwork**: [upwork.com/freelancers/~015fe5a704f8943797](https://www.upwork.com/freelancers/~015fe5a704f8943797)
- **Email**: nathanielgordon346@gmail.com
- **Location**: Tallahassee, FL, USA
