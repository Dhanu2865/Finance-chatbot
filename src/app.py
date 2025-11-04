
import streamlit as st
from dotenv import load_dotenv
from src import rag

load_dotenv()

st.set_page_config(
    page_title="💬 FinBot - Financial Literacy Chatbot",
    page_icon="💰",
    layout="centered",
)

st.markdown("""
<style>
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding-right: 8px;
}
.user-bubble {
    background-color: #DCF8C6;
    text-align: right;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}
.bot-bubble {
    background-color: #F1F0F0;
    text-align: left;
    padding: 12px;
    border-radius: 12px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align:center;'>💰 FinBot</h1>
<p style='text-align:center; color:gray;'>
Your AI-powered <b>Financial Literacy Chatbot</b><br>
Ask anything about RBI, SEBI, or IRDAI policies — for educational purposes only.
</p>
""", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = [
        {"role": "bot", "message": "Hello 👋 I'm FinBot! Ask me anything about financial literacy, RBI, SEBI, or IRDAI policies."}
    ]

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for chat in st.session_state.history:
    role, message = chat["role"], chat["message"]
    if role == "user":
        st.markdown(f"<div class='user-bubble'><b>You:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'><b>FinBot:</b> {message}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)


user_input = st.chat_input("💭 Type your question here...")

if user_input:

    st.session_state.history.append({"role": "user", "message": user_input})
    st.session_state.conversation_memory.append({"role": "user", "message": user_input})

    with st.spinner("🤖 FinBot is thinking..."):
        response = rag.generate_answer(user_input, st.session_state.conversation_memory)
        bot_reply = response.get("answer", "Sorry, I couldn’t process that right now.")
        sources = response.get("sources", [])

    st.session_state.history.append({"role": "bot", "message": bot_reply})
    st.session_state.conversation_memory.append({"role": "bot", "message": bot_reply})

    st.markdown(f"<div class='bot-bubble'><b>FinBot:</b> {bot_reply}</div>", unsafe_allow_html=True)

    if sources:
        st.caption("📄 Sources: " + ", ".join(sources))

st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
💡 <i>FinBot provides financial literacy insights only — not investment advice.</i><br>
⚙️ Powered by Gemini + RAG | Context-aware Conversation 🧠
</p>
""", unsafe_allow_html=True)
