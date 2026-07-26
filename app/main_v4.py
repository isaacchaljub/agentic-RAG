"""Streamlit chat UI for the fully agentic RAG pipeline (talks to serving_api/main_v4.py).

Multi-turn conversation without the caller juggling ids: the conversation_id returned by the
first /chat call is stashed in st.session_state and sent back on every later turn. st.session_state
persists across Streamlit's per-interaction reruns but resets when the browser tab is closed and
reopened, so a fresh visit starts a fresh conversation — the "close the page = new task" behaviour.

Run the API first, then this UI:
    fastapi run serving_api/main_v4.py --host 0.0.0.0 --port 8000
    streamlit run app/main_v4.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG — Chat", page_icon="💬")
st.title("Agentic RAG — Chat")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.caption(f"API: {API_URL}")
    if st.session_state.conversation_id:
        st.caption(f"Conversation: {st.session_state.conversation_id[:8]}…")
    if st.button("New conversation"):
        st.session_state.conversation_id = None
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"), st.spinner("Thinking…"):
        payload = {"message": prompt}
        if st.session_state.conversation_id:
            payload["conversation_id"] = st.session_state.conversation_id
        try:
            response = requests.post(f"{API_URL}/chat", json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            st.session_state.conversation_id = data["conversation_id"]
            answer = data["answer"]
        except Exception as e:
            answer = f"Error talking to the API at {API_URL}: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
