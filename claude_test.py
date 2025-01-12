
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Function to generate a response
def query_model(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send to GPU if available
    outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
import streamlit as st

# LLM query function (use the one defined earlier)
def query_model(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app layout
st.title("Open-Source LLM in Streamlit")
st.write("Ask anything, and the model will respond!")

# Input field
user_input = st.text_input("Enter your prompt:")

if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner("Generating response..."):
            response = query_model(user_input)
        st.success("Done!")
        st.write("### Response:")
        st.write(response)
    else:
        st.error("Please enter a prompt.")
