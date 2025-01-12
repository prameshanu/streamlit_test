import streamlit as st



pine_cone_api_key = st.secrets["PINE_CONE_API_KEY"]
claude_api_key = st.secrets["CLAUDE_API_KEY"]

langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]
hugging_face_api_key = st.secrets["HUGGING_FACE_API_KEY"]


st.title('Test')
st.write('PINE_CONE_API: ' , pine_cone_api_key)

