import os

from openai import OpenAI


def _join_double_newline(context):
    return "\n\n".join(context)


def _join_single_newline(context):
    return "\n".join(context)


def _join_bullet(context):
    return "\n".join(f"- {chunk}" for chunk in context)


def _join_space(context):
    return " ".join(context)


CONTEXT_JOINERS = {
    "double_newline": _join_double_newline,
    "single_newline": _join_single_newline,
    "bullet": _join_bullet,
    "space": _join_space,
}


def _prompt_template(query, context, prompt_template):
    return prompt_template.format(context=context, query=query)


def _prompt_simple(query, context, _prompt_template):
    return f"Context:\n{context}\n\nQuestion:\n{query}"


def _prompt_context_only(_query, context, _prompt_template):
    return context


def _prompt_question_only(query, _context, _prompt_template):
    return query


PROMPT_BUILDERS = {
    "template": _prompt_template,
    "simple": _prompt_simple,
    "context_only": _prompt_context_only,
    "question_only": _prompt_question_only,
}


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

    def _join_context(self, context, joiner):
        join_fn = CONTEXT_JOINERS.get(joiner, _join_double_newline)
        return join_fn(context)

    def _build_prompt(self, query, context, prompt_template, builder, context_joiner):
        joined_context = self._join_context(context, context_joiner)
        builder_fn = PROMPT_BUILDERS.get(builder, _prompt_template)
        return builder_fn(query, joined_context, prompt_template)

    def rewrite_query(self, query, rewrite_prompt_template, max_tokens=32, temperature=0.0):
        prompt = rewrite_prompt_template.format(query=query)

        if self.client is None:
            return query

        rewritten = self._chat_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        concise = (rewritten or "").strip()
        first_line = concise.split("\n", 1)[0].strip()
        return first_line or query

    def generate(
        self,
        query,
        context,
        prompt_template,
        max_tokens=300,
        temperature=0.2,
        prompt_builder="template",
        context_joiner="double_newline",
    ):
        prompt = self._build_prompt(
            query=query,
            context=context,
            prompt_template=prompt_template,
            builder=prompt_builder,
            context_joiner=context_joiner,
        )

        if self.client is None:
            return f"Mock answer for '{query}' with no OPENAI_API_KEY set."

        response = self._chat_complete(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (response or "").strip()[:max_tokens]
