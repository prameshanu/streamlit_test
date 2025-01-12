from transformers import AutoModelForCausalLM, AutoTokenizer
import streamlit as st

# Load model and tokenizer
model_name = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Define Streamlit app
st.title("Falcon-7B Streamlit App")
st.write("Ask a question, and Falcon-7B will answer!")

prompt = st.text_area("Your prompt:")
if st.button("Generate"):
    with st.spinner("Generating response..."):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.success("Response generated!")
    st.write(response)
