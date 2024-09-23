## Watch dog module to be used for docker
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
#beautiful soup for extracting info from website
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores  import Chroma


def splitWebIntoChunks(url):
    #get text in document format
    loader = WebBaseLoader(url)
    document = loader.load()
    
    #split text into chunks
    text_splitter = RecursiveCharacterTextSplitter()  ## can use SemanticChunker for splitting into sentences.
    documents = text_splitter.split_documents(document)

    #vectorstore for chunks
    vectorstore = Chroma.from_documents(documents, )

    return documents

def getResponse(query):
    return "I don't know"

#app config
st.set_page_config(page_title="Chat with Website", page_icon="<>")
st.title("Chat with website")

#side bar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website url")

#does not change eveytime the code is re-run
if "history" not in st.session_state:
    #chat history
    st.session_state.history = [
        AIMessage(content="Hello, how are you?"),
    ]

if website_url is None or website_url == "":
    st.info("Please enter a valid URL")
else:
    documents = splitWebIntoChunks(website_url)

    with st.sidebar:
        st.write(documents)

    #User query
    user_input = st.chat_input("Type your message here")
    if user_input is not None and user_input != "":
        response = getResponse(user_input)

        st.session_state.history.append(HumanMessage(content=user_input))
        st.session_state.history.append(AIMessage(content=response))

        # with st.chat_message("Human"):
        #     st.write(user_input)

        # with st.chat_message("AI"):
        #     st.write(response)

    #conversation - loop for all messages in the chat history - display messages acc to people wrote it 
    for message in st.session_state.history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)


