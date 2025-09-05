# Chapter 5: Building with DeepSeek

This chapter demonstrates how to build applications using DeepSeek models through various interfaces and frameworks.

## Installation

Install dependencies using uv:

```bash
uv sync
```

## Prerequisites

- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** - Python package manager
- **[Docker](https://docs.docker.com/get-docker/)** - For containerized deployments  
- **[Ollama](https://ollama.com/)** - For local model inference (example 06)
- **AWS Account** - Required for SageMaker examples (07-*)

> [!IMPORTANT]
> You need a Garmin Connect account to run this application.
> If you don't have one, set your `GARMIN_EMAIL=test@test.com` in `.envrc` to use sample data.

## File Overview

| File | Description |
|------|-------------|
| `01-initial-prototype.ipynb` | Jupyter notebook developing the initial health tracking prototype with Garmin data integration |
| `02-api.py` | FastAPI application for health summary API using DeepSeek models with structured JSON output |
| `02-api.Dockerfile` | Docker configuration for containerizing the FastAPI health summary API |
| `03-litellm.py` | Example using LiteLLM library to interface with DeepSeek models through multiple providers |
| `04-cpu-inference.py` | Local CPU inference example using Transformers library with DeepSeek-R1-Distill models |
| `05-api-cpu-xgrammar.py` | FastAPI application using local CPU inference with XGrammar for structured output generation |
| `05-api-cpu-xgrammar.Dockerfile` | Docker configuration for the local CPU inference FastAPI application |
| `06-ollama.py` | Integration with Ollama for local DeepSeek model inference and health data analysis |
| `07-api-deepseek-sagemaker.py` | FastAPI application using AWS SageMaker-deployed DeepSeek models for health summaries |
| `07-api-deepseek-sagemaker.Dockerfile` | Docker configuration for the AWS SageMaker FastAPI application |
| `07-aws-deployment.ipynb` | Jupyter notebook demonstrating AWS SageMaker deployment of DeepSeek models with structured output |
| `utils.py` | Utility functions including Garmin client setup, data processing, and Pydantic models for health data |

## Environment Variables

Copy the `.envrc` file in this directory and fill in your API keys and configuration values.

## Running Examples

### Python Scripts
Execute any script using uv:

```bash
uv run script_name.py
```

For example:
```bash
uv run 06-ollama.py
```

### FastAPI Applications
Run API applications using:

```bash
uv run fastapi 02-api.py
uv run fastapi 05-api-cpu-xgrammar.py  
uv run fastapi 07-api-deepseek-sagemaker.py
```

### Docker Deployments

#### Build Docker Images
```bash
# API with DeepSeek models
docker build -f 02-api.Dockerfile -t health-api .

# Local CPU inference API  
docker build -f 05-api-cpu-xgrammar.Dockerfile -t health-api-cpu .

# SageMaker API
docker build -f 07-api-deepseek-sagemaker.Dockerfile -t health-api-sagemaker .
```

#### Run Docker Containers
```bash
# API with DeepSeek models (requires DEEPSEEK_API_KEY)
docker run --env-file .envrc -p 8000:8000 health-api

# Local CPU inference API (no external API keys needed)
docker run -p 8000:8000 health-api-cpu

# SageMaker API (requires AWS credentials)
docker run --env-file .envrc -p 8000:8000 health-api-sagemaker
```

