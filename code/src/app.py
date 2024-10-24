from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma  # Chroma vectorstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
import streamlit as st

load_dotenv()  # make all the env variables available

# Function to get a response (placeholder)
def get_response(user_input):
    return "I don't know"

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(document)

    # vector store - Chroma
    vector_store = Chroma.from_documents(documents, OpenAIEmbeddings())

    return vector_store

# retriever chain - query embedding and context of the conversation
def retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    # chunks of text relevant to the entire conversation
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt)

    return history_aware_retriever

# Streamlit app configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("TalkNet")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    vector_store = get_vectorstore_from_url(website_url)
    context_retriever_chain = retriever_chain(vector_store)

    # User input
    user_query = st.text_input("Type your message here...")  

    # Handling user input and response
    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Call the retriever chain
        retrieved_documents = context_retriever_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_query
        })

        st.write(retrieved_documents)

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.expander("AI"):  # Assuming expander is used as a placeholder for chat_message
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.expander("Human"):  # Assuming expander is used as a placeholder for chat_message
                st.write(message.content)
