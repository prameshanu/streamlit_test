import streamlit as st
import anthropic

# Claude API Key (keep this secure!)
API_KEY = "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"

# Initialize the Claude client
client = anthropic.Client(api_key=API_KEY)

# Function to query Claude model
def query_claude(prompt, max_tokens=300):
    try:
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": f"{prompt}"}
            ]            
            )
        return response.get("completion", "No response received from Claude.")
    except Exception as e:
        return f"Error: {e}"

# Streamlit app
st.title("Streamlit App with Claude")
st.write("This app uses the Claude model to generate responses. Enter a prompt below:")

# Input for user prompt
user_prompt = st.text_area("Enter your prompt here:", height=150)

if st.button("Generate Response"):
    if user_prompt.strip():
        with st.spinner("Querying Claude..."):
            response = query_claude(user_prompt)
        st.success("Response received!")
        st.write("### Claude's Response:")
        st.write(response)
    else:
        st.error("Please enter a prompt before submitting!")
