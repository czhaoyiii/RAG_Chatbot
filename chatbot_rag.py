# Import basic
import os
from dotenv import load_dotenv

# Import Streamlit
import streamlit as st

# Import Pinecone
from pinecone import Pinecone

# Import LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

st.title("Chatbot")

# Initialise Pinecone database
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Set Pinecone index
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialise embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
# Initialise vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks."))

# Display chat history on chatbot
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

prompt = st.chat_input("How are you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Initialise LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5}
    )

    # Retrieve the documents related to the user's prompt
    docs = retriever.invoke(prompt)
    docs_text = "".join(d.page_content for d in docs)

    # LLM Prompt
    system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    Context: {context}:"""

    # Include the retrieved documents to the LLM prompt
    system_prompt_fmt = system_prompt.format(context=docs_text)

    print(system_prompt_fmt)

    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    result = llm.invoke(st.session_state.messages).content

    # Adding the chatbot response to the screen
    with st.chat_message("assistant"):
        st.markdown(result)
        st.session_state.messages.append(AIMessage(result))