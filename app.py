
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


st.set_page_config(page_title="AI Chat App")
st.title("🤖 Ask me anything!")


model_choice = st.selectbox("Choose a model:", [
    "qwen/qwen1.5-7b-chat",
    "llama3-8b-8192"
], index=0)


user_question = st.text_input("Enter your question:")


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer user queries in a simple and clear way."),
    ("user", "Question: {question}")
])


llm = ChatGroq(model=model_choice)


output_parser = StrOutputParser()


chain = prompt | llm | output_parser


if user_question:
    with st.spinner("Thinking..."):
        response = chain.invoke({"question": user_question})
        st.success("Response:")
        st.write(response)
