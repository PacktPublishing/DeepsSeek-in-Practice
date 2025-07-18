# Chapter 5: Building with DeepSeek

This chapter demonstrates how to build applications using DeepSeek models through various interfaces and frameworks.

## Prerequisites

### Install uv Package Manager

Install uv following the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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
uv run 06-llama-cpp-python.py
```

