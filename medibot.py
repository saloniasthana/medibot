import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    for _key in ("HF_TOKEN", "DATABASE_URL"):
        if _key not in os.environ and _key in st.secrets:
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

import database

DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""


@st.cache_resource
def ensure_db_ready():
    database.init_db()
    return True


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def load_llm(huggingface_repo_id, hf_token):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="conversational",
        temperature=0.5,
        huggingfacehub_api_token=hf_token,
        max_new_tokens=512,
    )
    return ChatHuggingFace(llm=llm)


def render_sources(source_documents):
    with st.expander("Source Documents"):
        for doc in source_documents:
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**{source}** (page {page})")
            st.caption(doc.page_content[:300] + "...")


def start_new_chat():
    st.session_state.current_session_id = (
        database.new_session_id() if st.session_state.user_id else None
    )
    st.session_state.messages = []


def render_login_signup():
    st.subheader("🔐 Welcome to Medibot")
    st.caption("Log in to save your chat history, or continue without an account.")

    with st.container(border=True):
        tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

        with tab_login:
            with st.form("login_form", border=False):
                username = st.text_input("Username", key="login_username", placeholder="your_username")
                password = st.text_input(
                    "Password", type="password", key="login_password", placeholder="••••••••"
                )
                if st.form_submit_button("Login", use_container_width=True):
                    try:
                        user_id = database.verify_login(username, password)
                    except Exception as e:
                        st.error(f"Could not reach the database: {e}")
                        user_id = None

                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.username = username
                        start_new_chat()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_signup:
            with st.form("signup_form", border=False):
                new_username = st.text_input(
                    "Choose a username", key="signup_username", placeholder="your_username"
                )
                new_password = st.text_input(
                    "Choose a password", type="password", key="signup_password", placeholder="••••••••"
                )
                if st.form_submit_button("Create Account", use_container_width=True):
                    if not new_username or not new_password:
                        st.error("Username and password are both required.")
                    else:
                        try:
                            ok, msg = database.create_user(new_username, new_password)
                        except Exception as e:
                            ok, msg = False, f"Could not reach the database: {e}"
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)

    st.caption("— or —")
    if st.button("Continue as Guest", use_container_width=True):
        st.session_state.guest = True
        start_new_chat()
        st.rerun()


def render_sidebar():
    with st.sidebar:
        st.header("About Medibot")
        st.markdown(
            "Medibot answers medical questions using *The Gale Encyclopedia of "
            "Medicine* as its knowledge base. It only answers from that source "
            "and won't make things up if it doesn't know."
        )
        st.divider()

        if st.session_state.user_id or st.session_state.guest:
            if st.button("+ New Chat", use_container_width=True):
                start_new_chat()
                st.rerun()

        if st.session_state.user_id:
            try:
                sessions = database.list_user_sessions(st.session_state.user_id)
            except Exception as e:
                sessions = []
                st.warning(f"Couldn't load chat history: {e}")

            if sessions:
                st.caption("CHATS")
                for s in sessions:
                    title = (s["title"] or "New conversation").strip()
                    label = (title[:28] + "…") if len(title) > 28 else title
                    is_active = s["session_id"] == st.session_state.current_session_id
                    if st.button(
                        ("▸ " if is_active else "   ") + label,
                        key=f"hist_{s['session_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.current_session_id = s["session_id"]
                        st.session_state.messages = database.get_session_messages(s["session_id"])
                        st.rerun()

        if st.session_state.user_id or st.session_state.guest:
            st.divider()

        if st.session_state.user_id:
            with st.container(border=True):
                st.markdown(f"🧑 **{st.session_state.username}**")
                st.caption("Chat history is saved to your account.")
                if st.button("Logout", use_container_width=True):
                    st.session_state.user_id = None
                    st.session_state.username = None
                    st.session_state.guest = False
                    st.session_state.current_session_id = None
                    st.session_state.messages = []
                    st.rerun()
        elif st.session_state.guest:
            with st.container(border=True):
                st.markdown("👤 **Guest Mode**")
                st.caption("This chat won't be saved.")
                if st.button("Log in", use_container_width=True):
                    st.session_state.guest = False
                    st.session_state.messages = []
                    st.rerun()
        else:
            render_login_signup()


def render_chat_area():
    st.title("🩺 Ask Medibot!")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Pass your prompt here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        if st.session_state.user_id:
            try:
                database.save_message(
                    st.session_state.user_id, st.session_state.current_session_id, "user", prompt
                )
            except Exception as e:
                st.sidebar.warning(f"Couldn't save message to history: {e}")

        try:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                st.error("HF_TOKEN not found. Please set it in your .env file.")
                return

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    vectorstore = get_vectorstore()

                    qa_chain = RetrievalQA.from_chain_type(
                        llm=load_llm(HUGGINGFACE_REPO_ID, hf_token),
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
                    )

                    response = qa_chain.invoke({"query": prompt})
                    result = response["result"]
                    source_documents = response["source_documents"]

                st.markdown(result)
                render_sources(source_documents)

            st.session_state.messages.append(
                {"role": "assistant", "content": result, "sources": source_documents}
            )
            if st.session_state.user_id:
                try:
                    database.save_message(
                        st.session_state.user_id, st.session_state.current_session_id, "assistant", result
                    )
                except Exception as e:
                    st.sidebar.warning(f"Couldn't save message to history: {e}")

        except Exception as e:
            st.error(f"Error: {str(e)}")


def main():
    st.set_page_config(page_title="Medibot", page_icon="\U0001FA7A", layout="wide")
    ensure_db_ready()

    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "guest" not in st.session_state:
        st.session_state.guest = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if not st.session_state.user_id and not st.session_state.guest:
        render_sidebar()
        st.title("🩺 Ask Medibot!")
        st.info("Log in, sign up, or continue as guest from the sidebar to start chatting.")
        return

    render_chat_area()
    render_sidebar()


if __name__ == "__main__":
    main()
