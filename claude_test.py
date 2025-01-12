

import streamlit as st


# Streamlit Framework
st.title('Claude TEST API')

import anthropic

# Define the API URL and key
# api_url = "https://api.anthropic.com/v1/completions"  # Confirm with Anthropic if this is the latest endpoint
api_url = "https://api.anthropic.com/v1/messages"

api_key = "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"  

import anthropic

# Initialize the Anthropic client with the correct API key
client = anthropic.Client(api_key=api_key)

# Define the prompt for Claude
prompt = "What are the benefits of exercise?"

# try:
#     # Query Claude model
#     response = client.messages.create(prompt=prompt, max_tokens=1024, model="claude-3-5-sonnet-20241022")
#     st.write(response['text'])
# except anthropic.AuthenticationError as e:
#     st.write(f"Authentication failed: {e}")
# except Exception as e:
#     st.write(f"Error occurred: {e}")

class ClaudeLLM:

        def query( prompt, max_tokens=1024):
                # Ensure the prompt starts with the correct conversational structure
                message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=max_tokens,
                messages=[
                {"role": "user", "content": f"{prompt}"}
                ]            
                )
                # Iterate over the list and extract text from each TextBlock
                extracted_texts = [block.text for block in message.content]
                
                return extracted_texts

        
# # Initialize Claude LLM
# api_key = my_api_key
llm = ClaudeLLM(api_key)

# prompt = """
# what is olympic games
# """

a = llm.query(prompt)
st.write(a)
