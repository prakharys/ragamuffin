class Generator:
    def __init__(self, llm_model):
        self.llm_model = llm_model

    def generate(self, query, context, prompt_template, max_tokens=300):
        prompt = prompt_template.format(
            context="\n\n".join(context),
            query=query
        )

        # Replace with real LLM request (OpenAI, Bedrock, local server, etc.).
        # Keep output bounded to mimic token control.
        answer = f"Mock answer for '{query}' using {self.llm_model}."
        return answer[:max_tokens]
