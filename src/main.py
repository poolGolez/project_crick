import streamlit as st

from src.qa_llm import ask

st.set_page_config(page_title="AWS Chatbot")
st.title("AWS Reference Chatbot")

if "ai_history" not in st.session_state and \
        "user_history" not in st.session_state and \
        "chat_history" not in st.session_state:
    print("Initializing states...")
    st.session_state["ai_history"] = []
    st.session_state["user_history"] = []
    st.session_state["chat_history"] = []

for user, message in st.session_state["chat_history"]:
    with st.chat_message(user):
        st.write(message)

if prompt := st.chat_input("What's your question?"):
    st.session_state["user_history"].append(prompt)
    st.session_state["chat_history"].append(("user", prompt))
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        result = ask(prompt, st.session_state["chat_history"])
        answer = result["answer"]
        with st.chat_message("ai"):
            st.write(answer)

    st.session_state["ai_history"].append(answer)
    st.session_state["chat_history"].append(("ai", answer))
