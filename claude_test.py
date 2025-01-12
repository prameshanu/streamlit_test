import streamlit as st


import requests



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
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
        st.write(answer)
    else:
        st.write("No 'Answer:' found in the generated text.")
else:
    st.write("Unexpected response format:", output)
