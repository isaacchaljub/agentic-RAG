import streamlit as st
import requests

#Create the Streamlit app

URL = "http://localhost:8000"

st.title("Agentic RAG")

st.subheader("Welcome to the Agentic RAG app")

def check_health():
        try:
            response = requests.get(URL + "/health")
            if response.status_code == 200:
                st.success("Health check passed")
            else:
                st.error("Health check failed")
        except Exception as e:
            st.error(f"Error checking health: {e}")

if st.button("Check Health", key="health_button"):
    check_health()

def answer_query(query):
    try:
        response = requests.post(URL + "/query", json={"query": query})
        if response.status_code == 200:
            st.write(response.json()["answer"])
        else:
            st.error(f"Error answering query: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error answering query: {e}")

# query = st.text_input("Enter your query here", key="query")

# if st.button("Answer Query", key="answer_button"):
#     if query:
#         with st.spinner("Processing your query..."):
#             answer_query(query)
#     else:
#         st.error("Please enter a query")
    
# Replace the button logic with a form:
with st.form("query_form", clear_on_submit=False):
    query = st.text_input("Enter your query here", key="query_form_input")
    submitted = st.form_submit_button("Answer Query")
    
    if submitted:
        if query:
            with st.spinner("Processing your query..."):
                answer_query(query)
        else:
            st.error("Please enter a query")
