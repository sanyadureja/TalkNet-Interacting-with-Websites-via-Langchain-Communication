import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
## Watch dog module to be used for docker


# RAG Class - chunking, embedding models
class ResponceLogic:
    def __init__(self) -> None:
        pass
    
    pass


#Application settings
class AppUI:
    def __init__(self) -> None:
        #app config
        st.set_page_config(page_title="Chat with Website", page_icon="<>")
        st.title("Chat with website")

        #side bar
        with st.sidebar:
            st.header("Settings")
            self.website_url = st.text_input("Website url")

        #does not change eveytime the code is re-run
        if "history" not in st.session_state:
            #chat history
            st.session_state.history = [
                AIMessage(content="Hello, how are you?"),
            ]
        
    def getResponse(self, query):
        return "I don't know"

    def run(self):
        if self.website_url is None or self.website_url == "":
            st.info("Please enter a valid URL")
        else:
            #User query
            user_input = st.chat_input("Type your message here")
            if user_input is not None and user_input != "":
                response = self.getResponse(user_input)

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


if __name__ == "__main__":
    app = AppUI()
    app.run()