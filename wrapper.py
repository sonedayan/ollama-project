# wrapper.py
import tiktoken
import os
from openai import OpenAI
from typing import List, Dict, Generator, Any

from dotenv import load_dotenv

load_dotenv()

class OllamaChat:
    def __init__(self, model: str = "gpt-oss:120b-cloud"):
        self.model = model
        self.client = OpenAI(
            base_url="https://ollama.com/v1",
            api_key=os.getenv('OLLAMA_API_KEY') # anything works
        )
        # Safe tokenizer for all Ollama models
        try:
            self.encoding = tiktoken.encoding_for_model(model.split(":")[0])
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: List[Dict]) -> int:
        tokens = 0
        for msg in messages:
            tokens += 4
            tokens += len(self.encoding.encode(msg["content"]))
        tokens += 2
        return tokens

    def stream_completion(self, messages: List[Dict[str, Any]]) -> Generator[str, None, Dict]:
        input_tokens = self.count_tokens(messages)
        collected = ""

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            stream=True
        )

        for chunk in stream:
            if content := chunk.choices[0].delta.content:
                collected += content
                yield content

        # Final stats
        yield {
            "input_tokens": input_tokens,
            "output_tokens": len(self.encoding.encode(collected)),
            "total_tokens": input_tokens + len(self.encoding.encode(collected)),
            "full_text": collected
        }