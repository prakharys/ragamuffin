import os

from openai import OpenAI


class Generator:
    def __init__(self, llm_model):
        self.llm_model = llm_model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

    def _chat_complete(self, messages, max_tokens, temperature=0.2):
        if self.client is None:
            return None

        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    def rewrite_query(self, query, rewrite_prompt_template, max_tokens=32):
        prompt = rewrite_prompt_template.format(query=query)

        if self.client is None:
            return query

        rewritten = self._chat_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,
        )

        concise = (rewritten or "").strip()
        first_line = concise.split("\n", 1)[0].strip()
        return first_line or query

    def generate(self, query, context, prompt_template, max_tokens=300):
        prompt = prompt_template.format(
            context="\n\n".join(context),
            query=query
        )

        if self.client is None:
            return f"Mock answer for '{query}' with no OPENAI_API_KEY set."

        response = self._chat_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return (response or "").strip()[:max_tokens]
