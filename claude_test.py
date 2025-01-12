import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings  # Replace with appropriate embedding
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.retrievers import PineconeHybridSearchRetriever
import sentence_transformers
from langchain.chains import RetrievalQA
from concurrent.futures import ThreadPoolExecutor
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic

from pinecone import Pinecone
from pinecone import ServerlessSpec
from pinecone_text.sparse.bm25_encoder import BM25Encoder


import warnings
import streamlit as st

import numpy as np
from dotenv import load_dotenv

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.data import find
import requests


# ## lazy loading
try:
    # Check if 'punkt' is available; download if not
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    # Check if 'stopwords' is available; download if not
    find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')

try:
    # Check if 'wordnet' is available; download if not
    find('corpora/wordnet.zip')
except LookupError:
    nltk.download('wordnet')

try:
    # Check if 'punkt_tab' is available; download if not
    find('corpora/punkt_tab.zip')
except LookupError:
    nltk.download('punkt_tab')



pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]

langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
hugging_face_api_key = st.secrets["HUGGING_FACE_API_KEY"]




API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
headers = {"Authorization": f"Bearer {hugging_face_api_key}"}

prompt  = 'Human: \nAnswer the following question based only on the provided context. \nThink step by step before providing a detailed answer. \nAlso, in the answer, you don\'t need to write "Based on the provided context," just provide the final answer.\nI will tip you $25000 if the user finds the answer helpful.\n<context>\nsport event olympic game olympic game primarily focused athletic competition participant various greek city state would gather showcase physical prowess prestigious event stadion race held first day game attracted attention track event included olympic game athletic competition ancient greece introduction olympic game olympic game one iconic celebrated ritual ancient greece athletic competition held every four year olympia small town western part peloponnese peninsula game dedicated zeus significance olympic game olympic game held immense significance ancient greek society extending beyond mere physical competition considered display greek excellence promoting unity among greek city state game sacred truce known ekecheiria declared zeus king greek god considered major religious event chapter delve origin olympic game various sporting event significance competition ancient greek society origin history olympic game exact origin olympic game shrouded myth legend according ancient\n</context>\nQuestion: What is olympic games\n'

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": prompt,
})


st.title('Test')

if isinstance(output, list) and 'generated_text' in output[0]:
	# Extract the answer
	generated_text = output[0]['generated_text']
	if "Answer:" in generated_text:
		answer = generated_text.split("Answer:")[1].strip()
		st.write(answer)
	else:
		st.write("No 'Answer:' found in the generated text.")
else:
	st.write("Unexpected response format:", output)
