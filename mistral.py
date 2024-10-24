from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

load_dotenv()  # Make all the env variables available

# Initialize Ollama embeddings
ollama_embeddings = OllamaEmbeddings(model="mistral")

def get_response(user_input):
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(documents=document_chunks, embedding=ollama_embeddings)
    return vector_store

# Retriever chain - query embedding and context of the conversation
def get_retriever_chain(vector_store):
    llm = Ollama(model="mistral")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    # Chunks of text relevant to the entire conversation
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    
    llm = Ollama(model="mistral")
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# Streamlit app configuration
st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
st.title("TalkNet")

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello,how can I help you?")
    ]

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)  

    # Handling user input and response
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        # Placeholder for displaying retrieved documents or handling with the retrieval chain
        # This would be where you integrate more complex logic based on user_query and website content

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):  # Open a chat message block styled as "AI"
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):  # Open a chat message block styled as "Human"
                st.write(message.content)