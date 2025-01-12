import anthropic

import streamlit as st
# Streamlit Framework
st.title('Langchain Demo incorporating Hybrid Search With LLAMA2 API')


class MyLLM:
    def query(self, prompt):
        try:
            # Make the request to the API
            my_api_key = "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"  # Replace with your API key

            # Ensure that you have initialized the client correctly with a valid API key
            client = anthropic.Client(api_key=my_api_key)

            response = client.messages.create(model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]
                                             )
            return response
        except NameError as e:
            print(f"NameError: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

# Initialize your LLM object and call the query method
llm = MyLLM()
prompt = "What are the benefits of exercise?"
response = llm.query(prompt)

st.write(response)
