# Getting Started with DSPy

This directory contains examples to help you get started with DSPy.

## Quick Start (No API Key Required!)

**Run the dry-run example** to test DSPy without making real API calls:

```bash
python getting_started_dry_run.py
```

This uses `DummyLM` to simulate LLM responses, perfect for:
- Learning DSPy's API and concepts
- Testing your code structure
- Development without API costs
- Environments with restricted network access

## Examples Included

### 1. `getting_started_dry_run.py` - Testing Without API Calls ✨

Uses DSPy's built-in `DummyLM` for simulation. Demonstrates:
- **Basic Q&A**: Simple question answering
- **Chain of Thought**: Reasoning with step-by-step thinking
- **Dictionary Mode**: Context-aware responses
- **Custom Modules**: Creating your own DSPy components

**No setup required - just run it!**

### 2. `dspy_basic_example.py` - Real API Calls

Shows DSPy with actual OpenAI API integration. Requires:

```bash
export OPENAI_API_KEY='sk-proj-...'
python dspy_basic_example.py
```

Demonstrates:
- Basic question answering with `Predict`
- Chain of thought reasoning with `ChainOfThought`
- String signature shorthand
- Error handling and provider options

## Using Your API Key

The DSPy code structure is correct and your API key format looks valid! However, this environment has network restrictions that block external API calls (proxy 403 error).

**To use your API key in a local environment:**

```python
import dspy
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-proj-your-api-key-here"

# Initialize LM
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

# Use it!
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)
```

## Supported Providers

DSPy supports many LLM providers via LiteLLM:

```python
# OpenAI
lm = dspy.LM('openai/gpt-4o-mini')

# Anthropic Claude
lm = dspy.LM('anthropic/claude-3-5-sonnet-20241022')

# Google Gemini
lm = dspy.LM('google/gemini-pro')

# Azure OpenAI
lm = dspy.LM('azure/your-deployment-name')

# Local models (Ollama, vLLM, etc.)
lm = dspy.LM('ollama/llama2')
```

## Next Steps

1. **Start with the dry-run example** to learn DSPy's API
2. **Check out the [official docs](https://dspy.ai)** for advanced features
3. **Explore DSPy modules**: Retrieve, ChainOfThought, ReAct, etc.
4. **Try optimization**: BootstrapFewShot, MIPRO, etc.

## Common DSPy Patterns

### Signature Definition
```python
class MyTask(dspy.Signature):
    """Description of the task."""
    input_field: str = dspy.InputField()
    output_field: str = dspy.OutputField()
```

### Using Predictors
```python
# Simple prediction
predictor = dspy.Predict(MyTask)
result = predictor(input_field="...")

# Chain of thought
cot = dspy.ChainOfThought(MyTask)
result = cot(input_field="...")
```

### Custom Modules
```python
class MyModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict("input -> output")

    def forward(self, input):
        return self.predictor(input=input)
```

Happy building with DSPy! 🚀
