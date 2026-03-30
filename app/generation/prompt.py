def build_prompt(query, context):
    return f"""
You are an enterprise AI assistant.

Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{query}
"""