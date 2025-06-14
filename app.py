import os
import streamlit as st
from dotenv import load_dotenv
# Import the pipeline from main.py so we can delegate user queries
from main import run as process_query
# openai/OpenAI import no longer needed here since main.py handles any OpenAI calls
from openai import OpenAI

# Basic Streamlit page configuration
st.set_page_config(page_title="LLM Chat", page_icon="🧠")
st.title("🧠 LLM Chat")

# Initialize chat history in the session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Helper function to display chat messages stored in session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box for the user to type a new message
prompt = st.chat_input("Type your message and press Enter...")

# Sidebar switch for choosing agent mode
mode_choice = st.sidebar.radio(
    "Agent reasoning mode",
    options=["reAct", "planning"],
    index=0,
    format_func=lambda m: "ReAct (direct tool use)" if m == "reAct" else "Planning (plan then execute)"
)

if prompt:
    # Show the user's message in the chat UI immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add user's message to the session history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Placeholder for the assistant's response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        try:
            # Delegate the user prompt to main.py's run function
            response_text = process_query(prompt, mode=mode_choice.lower())

            # Render the response in the UI
            response_placeholder.markdown(response_text)

            # Save assistant response to the session history
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )
        except Exception as e:
            error_msg = f"Error fetching response: {e}"
            response_placeholder.markdown(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
