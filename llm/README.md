# LLM Annotation Module

## Overview

This module provides LLM-based text annotation using Qwen2.5-7B-Instruct with structured JSON output for classification tasks.

## Prompt Templates (`prompts.py`)

**Zero-shot prompts** with strict JSON schema enforcement:
- **Sentiment Analysis** (IMDb): positive/negative classification
- **Toxicity Detection** (Jigsaw): toxic/non-toxic classification
- **Fact Verification** (FEVER): SUPPORTS/REFUTES/NOT ENOUGH INFO three-way classification

Each prompt includes:
- Clear task description with label definitions
- Strict JSON output schema with `<json>` sentinel tags
- Verbal confidence levels (low/medium/high)
- Rationale requirements (10-200 char evidence from text)

**Key Design Decisions**:
- **Zero-shot only**: No few-shot examples to avoid label bias
- **Sentinel tags**: `<json>...</json>` for robust extraction
- **Explicit instructions**: "Output ONE JSON object and nothing else"

## Annotator (`annotator.py`)

**QwenAnnotator** with Qwen2.5-7B-Instruct:
- **Robust JSON extraction**: Multi-stage parsing with fallback strategies
- **Error classification**: Categorizes failures (parse_error_recoverable, schema_violation, etc.)
- **Batch processing**: Left-padding for decoder-only models
- **GPU optimization**: Automatic device mapping with bfloat16
- **Reproducibility**: Fixed seed (42) for deterministic results

**JSON Extraction Priority**:
1. From `<json></json>` tags (last occurrence)
2. From markdown code blocks
3. Last complete JSON object (tail-end extraction)
4. Fallback: first `{` to last `}`

## Model Configuration

**Model**: `Qwen/Qwen2.5-7B-Instruct` (7B parameters)

**Generation Settings** (see `config.py`):
- `temperature=0.1` - Low temperature for focused classification (no thinking mode)
- `max_tokens=128` - Sufficient for JSON output
- `top_p=0.8` - Standard nucleus sampling
- `top_k=20` - Diverse but focused outputs
- `do_sample=True` - Required for temperature-based sampling
- `repetition_penalty=1.05` - Prevents repetitive outputs
- `seed=42` - Fixed seed for reproducibility

## References

- [Qwen2.5-7B-Instruct Model Card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen2.5 Documentation](https://qwen.readthedocs.io/)
