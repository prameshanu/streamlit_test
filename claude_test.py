import anthropic

import streamlit as st
# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')


from transformers import pipeline

# GPT-J model
llm = pipeline("text-generation", model="EleutherAI/gpt-j-6B")

# Query the LLM
prompt = "What are the benefits of exercise?"
response = llm(prompt, max_length=100)
a = response[0]["generated_text"]

st.write(a)
