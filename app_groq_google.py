import streamlit as st
# from langchain_community.tools import WikipediaQueryRun
# from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import GoogleSearchResults
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
import os
import warnings
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------- TOOL SETUP ----------------------------
api_wrapper = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
    k=1
)
google = GoogleSearchResults(api_wrapper=api_wrapper)

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(page_title="Smart AI Assistant", page_icon="💡", layout="centered")

# ---------------------------- API KEY ----------------------------
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
api_key = os.getenv("GROQ_API_KEY")

# ---------------------------- CUSTOM CSS ----------------------------
st.markdown("""
    <style>
        .stApp {
            background-color: #fefefe;
            font-family: 'Segoe UI', sans-serif;
        }
        .chat-message {
            padding: 0.75rem;
            border-radius: 10px;
            margin: 10px 0;
            width: fit-content;
            max-width: 85%;
        }
        .user {
            background-color: #d9f4ff;
            margin-left: auto;
        }
        .assistant {
            background-color: #f3e5f5;
            margin-right: auto;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------- HEADER ----------------------------
st.markdown("<h1 style='text-align: center;'>💡 GHOSH & CHOWDHURY AI SOLUTIONS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>WELCOME TO THE AI WORLD</p>", unsafe_allow_html=True)

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.image("pic.png", width=150)
st.sidebar.title("🗂️Topics Discussed")

# ---------------------------- TOPIC MANAGEMENT ----------------------------
if "topic_summary" not in st.session_state:
    st.session_state["topic_summary"] = ""
if "last_topic_question" not in st.session_state:
    st.session_state["last_topic_question"] = ""
if "previous_topics" not in st.session_state:
    st.session_state["previous_topics"] = []

def summarize_topic(question: str):
    topic_prompt = f"Summarize the following question in 1-2 words:\n\n{question}\n\nTopic:"
    topic_llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key, temperature=0.7, max_tokens=200)
    raw = topic_llm([HumanMessage(content=topic_prompt)])
    summary = raw.content.strip().split("\n")[0].replace("Topic:", "").strip(" \"'")
    return summary

# ---------------------------- CHAT HISTORY ----------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hello! I am your assistant John. I'm here to help you."}
    ]

# ---------------------------- DISPLAY CHAT ----------------------------
for msg in st.session_state.messages:
    role_class = "user" if msg["role"] == "user" else "assistant"
    st.markdown(f"<div class='chat-message {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

# ---------------------------- CHAT INPUT ----------------------------
if prompt := st.chat_input("Ask me anything..."):
    st.session_state["messages"].append({"role": "user", "content": f"👤 {prompt}"})
    st.markdown(f"<div class='chat-message user'>👤 {prompt}</div>", unsafe_allow_html=True)

    # Update topic if new question
    if prompt != st.session_state["last_topic_question"]:
        topic = summarize_topic(prompt)
        st.session_state["topic_summary"] = topic
        st.session_state["last_topic_question"] = prompt
        if topic not in st.session_state["previous_topics"]:
            st.session_state["previous_topics"].insert(0, topic)  # insert at top

    # LLM for assistant response
    llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key, temperature=0.7, max_tokens=150)

    tools = [google]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True
    )

    with st.chat_message("assistant"):
        with st.spinner("🤖 Thinking..."):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)

            # Limit history to last 10 exchanges
            st.session_state["messages"] = st.session_state["messages"][-10:]

            # Reconstruct chat history as context
            chat_history_input = ""
            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    chat_history_input += f"User: {msg['content'].replace('👤 ', '')}\n"
                elif msg["role"] == "assistant":
                    chat_history_input += f"Assistant: {msg['content'].replace('🤖 ', '')}\n"
            chat_history_input += f"User: {prompt}\nAssistant:"

            # Generate assistant response
            response = agent.run(chat_history_input, callbacks=[st_cb])
            response = f"🤖 {response}"
            st.session_state["messages"].append({"role": "assistant", "content": response})
            st.markdown(f"<div class='chat-message assistant'>{response}</div>", unsafe_allow_html=True)

# ---------------------------- SIDEBAR TOPIC DISPLAY ----------------------------
if st.session_state["topic_summary"]:
    st.sidebar.markdown(f"**🟢 {st.session_state['topic_summary']}**")
else:
    st.sidebar.markdown("_No topic yet. Ask a question!_")

if st.session_state["previous_topics"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📜 Previous Topics")
    for topic in st.session_state["previous_topics"]:
        st.sidebar.markdown(f"- {topic}")

st.sidebar.markdown("---")
st.sidebar.info("Ask a question below to start the session.")
