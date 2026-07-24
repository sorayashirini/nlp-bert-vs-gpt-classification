# NLP Text Classification: Fine-Tuned BERT vs Prompt-Based LLM

MSc Business Analytics coursework project, DCU.

## Overview

A structured model evaluation comparing two approaches to the same text
classification task: a fine-tuned BERT classifier and prompt-based
classification using an open-source LLM (Llama 3.1) served locally via
the Ollama API. The task is sentiment classification (positive /
neutral / negative) of 800,000+ Glassdoor employee reviews.

## What this project does

- Fine-tunes a BERT model (bert-base-uncased, Hugging Face
  Transformers) on the sentiment classification task
- Runs the same task through a prompt-based LLM approach: zero-shot
  classification with Llama 3.1 via a local Ollama API, evaluated on a
  stratified 10,000-review test subset
- Benchmarks both with per-class and weighted F1 scores rather than
  accuracy alone
- Performs a structured failure-mode analysis: where does each approach
  break down, and why
- Examines prompt sensitivity in the LLM-based approach

## Why it matters

Headline accuracy hides how models fail. This project documents the
failure modes of each approach - the kind of evaluation needed before
deploying either in production. Using a locally served open-source LLM
also makes the prompt-based baseline fully reproducible at zero API
cost.

## Methods and tools

Python, Hugging Face Transformers (PyTorch) for BERT fine-tuning with
WordPiece tokenization, pandas and numpy for data preparation, Ollama
(Llama 3.1) with the requests library for prompt-based classification,
and scikit-learn, matplotlib, and seaborn for evaluation and
visualisation.

## Results

BERT outperformed the prompt-based LLM on every metric: accuracy 0.76
vs 0.53 and weighted F1 0.75 vs 0.55, with per-class F1 of 0.71 vs
0.57 (negative), 0.48 vs 0.29 (neutral), and 0.87 vs 0.64 (positive).
The main failure-mode finding: both models broke down on neutral
sentiment - frequently misclassifying it as positive due to label
ambiguity - and the LLM's predictions were less stable overall,
varying with prompt design.

## Repository contents

- `notebook.ipynb` - full pipeline: data preparation, BERT
  fine-tuning, LLM prompt-based classification, and evaluation
- `nlp-sentiment-report.pdf` - full project report
- `README.md` - this file
