# Chapter 5: Building with DeepSeek

This chapter demonstrates how to build applications using DeepSeek models through various interfaces and frameworks.

## Installation

Install dependencies using uv:

```bash
uv sync
```

## File Overview

| File | Description |
|------|-------------|
| `01-initial-prototype.ipynb` | Jupyter notebook developing the initial health tracking prototype with Garmin data integration |
| `02-api.py` | FastAPI application for health summary API using DeepSeek models with structured JSON output |
| `03-litellm.py` | Example using LiteLLM library to interface with DeepSeek models through multiple providers |
| `04-cpu-inference.py` | Local CPU inference example using Transformers library with DeepSeek-R1-Distill models |
| `05-api-cpu-xgrammar.py` | FastAPI application using local CPU inference with XGrammar for structured output generation |
| `06-ollama.py` | Integration with Ollama for local DeepSeek model inference and health data analysis |
| `07-api-deepseek-sagemaker.py` | FastAPI application using AWS SageMaker-deployed DeepSeek models for health summaries |
| `07-aws-deployment.ipynb` | Jupyter notebook demonstrating AWS SageMaker deployment of DeepSeek models with structured output |
| `utils.py` | Utility functions including Garmin client setup, data processing, and Pydantic models for health data |

## Prerequisites

### Environment Variables

Copy the `.envrc` file in this directory and fill in your API keys and configuration values.

**Linux/macOS:**
```bash
export DEEPSEEK_API_KEY="your-api-key-here"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

**Windows (Command Prompt):**
```cmd
set DEEPSEEK_API_KEY=your-api-key-here
set DEEPSEEK_BASE_URL=https://api.deepseek.com
```

**Windows (PowerShell):**
```powershell
$env:DEEPSEEK_API_KEY="your-api-key-here"
$env:DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### Additional Tools

For local inference examples, install:
- **[Ollama](https://ollama.com/)**
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)**

## Running Examples

Execute any script using uv:

```bash
uv run script_name.py
```

For example:
```bash
uv run 06-ollama.py
```

