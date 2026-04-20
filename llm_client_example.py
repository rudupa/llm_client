"""
LLM API Example — Anthropic Claude / Google Gemini / Ollama (local)

Install dependencies:
    pip install anthropic google-genai ollama python-dotenv

Setup:
    Copy .env.example to .env and fill in keys for the provider(s) you want.
    For Ollama: install from https://ollama.com and run `ollama pull llama3.2`
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env file


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG — change PROVIDER to switch between backends
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER = "ollama"      # Options: "claude" | "gemini" | "ollama"

MODELS = {
    "claude": "claude-3-7-sonnet-20250219",
    "gemini": "gemini-2.0-flash",
    "ollama": "llama3.2",           # any model you have pulled locally
}

OLLAMA_BASE_URL = "http://localhost:11434"  # default Ollama address


# ─────────────────────────────────────────────────────────────────────────────
#  Unified LLM client
# ─────────────────────────────────────────────────────────────────────────────

class LLMResponse:
    """Normalized response returned by all providers."""

    def __init__(self, text: str, input_tokens: int = 0, output_tokens: int = 0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def __str__(self) -> str:
        return self.text


class LLMClient:
    """
    Thin wrapper that exposes a single chat() method regardless of provider.
    Supports: claude, gemini, ollama.
    """

    def __init__(self, provider: str = PROVIDER):
        self.provider = provider.lower()
        self.model = MODELS[self.provider]
        self._client = self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            from anthropic import Anthropic
            return Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

        if self.provider == "gemini":
            from google import genai
            return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        if self.provider == "ollama":
            import ollama as _ollama
            return _ollama

        raise ValueError(f"Unknown provider: {self.provider!r}")

    def chat(
        self,
        messages: list,
        system: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Send a chat request.

        messages: list of {"role": "user"|"assistant", "content": "..."}
        system:   optional system prompt string
        stream:   if True, prints tokens live and returns the full text
        """
        if self.provider == "claude":
            return self._chat_claude(messages, system, max_tokens, temperature, stream)
        if self.provider == "gemini":
            return self._chat_gemini(messages, system, max_tokens, temperature, stream)
        if self.provider == "ollama":
            return self._chat_ollama(messages, system, max_tokens, temperature, stream)

    # ── Claude ────────────────────────────────────────────────────────────────

    def _chat_claude(self, messages, system, max_tokens, temperature, stream):
        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        if stream:
            full_text = ""
            with self._client.messages.stream(**kwargs) as s:
                for chunk in s.text_stream:
                    print(chunk, end="", flush=True)
                    full_text += chunk
            print()
            return LLMResponse(full_text)

        resp = self._client.messages.create(**kwargs)
        return LLMResponse(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

    # ── Gemini ────────────────────────────────────────────────────────────────

    def _chat_gemini(self, messages, system, max_tokens, temperature, stream):
        from google.genai import types

        # Build Gemini content list
        contents = [
            types.Content(
                role="user" if m["role"] == "user" else "model",
                parts=[types.Part(text=m["content"])],
            )
            for m in messages
        ]

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system,
        )

        if stream:
            full_text = ""
            for chunk in self._client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                print(chunk.text, end="", flush=True)
                full_text += chunk.text
            print()
            return LLMResponse(full_text)

        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text,
            input_tokens=usage.prompt_token_count if usage else 0,
            output_tokens=usage.candidates_token_count if usage else 0,
        )

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _chat_ollama(self, messages, system, max_tokens, temperature, stream):
        import ollama

        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        options = {"temperature": temperature, "num_predict": max_tokens}

        if stream:
            full_text = ""
            for chunk in ollama.chat(
                model=self.model,
                messages=all_messages,
                stream=True,
                options=options,
            ):
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full_text += token
            print()
            return LLMResponse(full_text)

        resp = ollama.chat(model=self.model, messages=all_messages, options=options)
        return LLMResponse(
            text=resp["message"]["content"],
            input_tokens=resp.get("prompt_eval_count", 0),
            output_tokens=resp.get("eval_count", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Examples  (all use the unified LLMClient — provider set by CONFIG above)
# ─────────────────────────────────────────────────────────────────────────────

def basic_message():
    """Single-turn message."""
    client = LLMClient()
    resp = client.chat(
        messages=[{"role": "user", "content": "Explain what a REST API is in two sentences."}]
    )
    print(f"=== Basic Message [{PROVIDER}] ===")
    print(resp.text)
    print(f"Tokens — input: {resp.input_tokens}, output: {resp.output_tokens}\n")


def system_prompt_example():
    """Message with a system prompt."""
    client = LLMClient()
    resp = client.chat(
        system="You are a senior software architect. Answer concisely and use bullet points.",
        messages=[{"role": "user", "content": "What are the key principles of clean code?"}],
    )
    print(f"=== System Prompt [{PROVIDER}] ===")
    print(resp.text)
    print()


def multi_turn_conversation():
    """Multi-turn conversation."""
    client = LLMClient()
    history = []

    turns = [
        "What is dependency injection?",
        "Give me a simple Python example of that.",
        "How would I unit test that code?",
    ]

    print(f"=== Multi-Turn Conversation [{PROVIDER}] ===")
    for user_input in turns:
        history.append({"role": "user", "content": user_input})
        resp = client.chat(messages=history)
        history.append({"role": "assistant", "content": resp.text})

        print(f"User: {user_input}")
        print(f"AI:   {resp.text}\n")


def streaming_example():
    """Streaming response — prints tokens as they arrive."""
    client = LLMClient()
    print(f"=== Streaming [{PROVIDER}] ===")
    print("AI: ", end="", flush=True)
    client.chat(
        messages=[{"role": "user", "content": "Count from 1 to 10, one number per line."}],
        stream=True,
    )
    print()


def document_transform_example(markdown_content: str) -> str:
    """
    Transform a requirements Markdown document into a structured ISR document.
    Matches the use case described in DESIGN_DOCUMENT.md.
    """
    client = LLMClient()

    system_prompt = (
        "You are a senior software architect and technical writer. "
        "Transform the provided requirements document into a structured "
        "Internal System Requirements (ISR) document. "
        "Assign ISR-XXX IDs to each requirement, categorize them "
        "(Functional, Non-Functional, Security, Interface, Constraint), "
        "and assign priorities (Critical, High, Medium, Low). "
        "Output only Markdown."
    )

    resp = client.chat(
        system=system_prompt,
        max_tokens=4096,
        temperature=0.3,
        messages=[
            {
                "role": "user",
                "content": f"Transform the following requirements document into an ISR:\n\n{markdown_content}",
            }
        ],
    )
    return resp.text


def read_md_file_and_transform(file_path: str):
    """Read a .md requirements file, transform it, and save the ISR output."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"=== Document Transform: {file_path} [{PROVIDER}] ===")
    result = document_transform_example(content)
    print(result)

    output_path = file_path.replace(".md", "_ISR.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\nISR saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Provider: {PROVIDER}  |  Model: {MODELS[PROVIDER]}\n")

    basic_message()
    system_prompt_example()
    multi_turn_conversation()
    streaming_example()

    # Uncomment to transform a local .md requirements file:
    # read_md_file_and_transform("requirements.md")
