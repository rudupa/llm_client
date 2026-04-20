# LLM Client Example

A multi-provider Python LLM client that abstracts **Anthropic Claude**, **Google Gemini**, and **Ollama** (local, free, no API key) behind a single unified interface.

Includes a `DESIGN_DOCUMENT.md` describing a full React application that uses the same multi-provider pattern to transform Markdown requirements documents into structured Internal System Requirements (ISR) documents.

---

## Repository Contents

| File | Description |
|------|-------------|
| `llm_client_example.py` | Main Python script — unified `LLMClient` class + runnable examples |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for API keys |
| `DESIGN_DOCUMENT.md` | Architecture & design doc for the React Requirements Transformer app |

---

## Supported Providers

| Provider | Cost | API Key Required | Model |
|----------|------|-----------------|-------|
| **Ollama** (local) | Free | No | `llama3.2` (or any pulled model) |
| **Google Gemini** | Free tier available | Yes | `gemini-2.0-flash` |
| **Anthropic Claude** | Paid (no free tier) | Yes | `claude-3-7-sonnet-20250219` |

---

## Prerequisites

### Python
- Python 3.9 or higher
- Recommended: use a virtual environment

### Ollama (for local, free inference)
Download and install from **https://ollama.com**

**Windows (winget):**
```powershell
winget install Ollama.Ollama
```

After installation, Ollama starts automatically as a background service on `http://localhost:11434`.

Pull a model before running (one-time, ~2 GB download):
```powershell
ollama pull llama3.2
```

List available local models:
```powershell
ollama list
```

---

## Setup

### 1. Clone the repository
```powershell
git clone https://github.com/rudupa/llm_client.git
cd llm_client
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies
```powershell
pip install -r requirements.txt
```

> **Corporate / restricted network note:** If pip is configured to use an internal mirror that is unreachable, install directly from PyPI:
> ```powershell
> pip install -r requirements.txt --index-url https://pypi.org/simple/
> ```

### 4. Configure API keys (optional — not needed for Ollama)
```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill in the keys for any cloud provider you want to use:
```env
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

> **Ollama users:** No `.env` changes needed. Leave the keys blank.

---

## Switching Providers

Open `llm_client_example.py` and change the `PROVIDER` constant at the top:

```python
PROVIDER = "ollama"   # "ollama" | "gemini" | "claude"
```

Optionally change the model:
```python
MODELS = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",   # any model you have pulled
}
```

---

## Running the Script

```powershell
# With venv activated:
python llm_client_example.py

# Or directly (no activation needed):
.venv\Scripts\python.exe llm_client_example.py
```

The script runs four examples in sequence:

| Example | What it demonstrates |
|---------|---------------------|
| `basic_message()` | Single user message |
| `system_prompt_example()` | System prompt controlling AI persona |
| `multi_turn_conversation()` | Stateful back-and-forth chat |
| `streaming_example()` | Token-by-token streaming output |

### Transform a requirements `.md` file to ISR

Uncomment the last line in `llm_client_example.py`:
```python
read_md_file_and_transform("requirements.md")
```

Then run:
```powershell
python llm_client_example.py
```

Output is saved as `requirements_ISR.md` in the same directory.

---

## API Key Links

| Provider | Get a key |
|----------|-----------|
| Anthropic Claude | https://console.anthropic.com/ |
| Google Gemini | https://aistudio.google.com/app/apikey |
| Ollama | No key — runs locally |

---

## Related

See [`DESIGN_DOCUMENT.md`](DESIGN_DOCUMENT.md) for the full architecture design of the React-based Requirements Transformer web application that uses this same multi-provider pattern in TypeScript.
