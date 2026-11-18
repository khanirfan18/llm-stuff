
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.8,
    max_tokens=1000
)

st.title("Hinglish Roaster Bot")

if "message" not in st.session_state:
    st.session_state.message = [
        SystemMessage(
            content="You are funny roaster agent who roasts and answers in hinglish"
        )
    ]

# Display conversation
for msg in st.session_state.message[1:]:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)

    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    if prompt == "0":
        st.stop()

    st.session_state.message.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response = model.invoke(st.session_state.message)
        st.write(response.text)

    st.session_state.message.append(AIMessage(content=response.content))
