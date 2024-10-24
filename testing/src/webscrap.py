# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma  #Chroma vectorstore
# from Langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# import streamlit as st
 
# load_dotenv() #make all the env variables available

# # Function to get a response (placeholder)
# def get_response(user_input):
#     return "I don't know"

# def get_vectorstore_from_url(url):
#     loader=WebBaseLoader (url)
#     document=loader.load ()
#     #vector store-Chroma
#     # vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

#     return documents

# #retriever chain-query embedding and context of the conversation
# # def retriever_chain(vector_store):
# #     llm=ChatOpenAI()
# #     retriever = vector_store.as_retriever()
# #     #chunks of text relevant to the entire conversation
# #     prompt = ChatPromptTemplate.from_messages([
# #       MessagesPlaceholder(variable_name="chat_history"),
# #       ("user", "{input}"),
# #       ("user", "Generate a search query in order to get information relevant to the conversation")
# #     ])
    
# #     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
# #     return retriever_chain

# # Streamlit app configuration
# st.set_page_config(page_title="Chat with Websites", page_icon="🤖")
# st.title("TalkNet")

# # Initialize chat history in session state if not already present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         AIMessage(content="Hello, I am a bot. How can I help you?"),
#     ]

# # Sidebar settings
# with st.sidebar:
#     st.header("Settings")
#     website_url = st.text_input("Website URL")

# if website_url is None or website_url == "":
#     st.info("Please enter a website URL")
# else:
# #    vector_store= get_vectorstore_from_url(website_url)
# #    retriever_chain = get_context_retriever_chain(vector_store)
#     # with st. sidebar:
#     #     st.write(documents)

#     # User input
#     user_query = st.chat_input("Type your message here...")  # Assuming chat_input exists in your version of Streamlit

#     # Handling user input and response
#     if user_query is not None and user_query != "":
#         response = get_response(user_query)
#         st.session_state.chat_history.append(HumanMessage(content=user_query))
#         st.session_state.chat_history.append(AIMessage(content=response))

#         #call the retriever chain
#         # retrieved_documents = retriever_chain.invoke({
#         #     {"chat_history": st.session_state.chat_history,
#         #     "input":user_query
#         # })

#         # st.write(retrieved_documents)

#     # Display conversation
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):  # Open a chat message block styled as "AI"
#                 st.write(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):  # Open a chat message block styled as "Human"
#                 st.write(message.content)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma  # Chroma vectorstore, commented as it's not used in the corrected script
# from Langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
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
    return documents

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
    # Assuming the use of a placeholder function for demonstration
    # The actual functionality to use vector_store and retrieval should replace these comments
    documents = get_vectorstore_from_url(website_url)  # This line is just to show how you'd call the function, not used further in the script
    with st.sidebar:
        st.write(documents)
    # User input
    user_query = st.chat_input("Type your message here...")  # Corrected from st.chat_input to st.text_input

    # Handling user input and response
    if user_query:
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