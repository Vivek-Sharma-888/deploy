
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader,PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
import streamlit as st


if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def process_document(path):
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=30)
    splitted_text = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_stores = InMemoryVectorStore.from_documents(
        documents = splitted_text,
        embedding = embeddings
    )

    @tool
    def retrieval_tool(query:str):
        '''
        this tool to address multiple question in a single query
        '''
        data = vector_stores.similarity_search(query=query,k=2)

        context =" "
        for doc in data:
            context+= doc.page_content + '\n\n'

        return context

    llm = ChatGroq(model="openai/gpt-oss-20b")

    System_Prompt = """
        You are a helpful assistant that answers questions using retrieved context.
        ALWAYS use the `retriever_tool` tool for questions requiring external knowledge.
    """

    agent = create_agent(
        model = llm,
        tools = [retrieval_tool],
        system_prompt = System_Prompt,
        checkpointer = InMemorySaver()
    )

    st.session_state.agent = agent
    st.session_state.document_uploaded = True

# query = "Name of client and Address of contractor"
# response = agent.invoke({"messages":[{"role":"user", "content":query}]})
# result = response["messages"][-1].content
# print(result)


if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF files",type=["pdf"],accept_multiple_files=True)
    if uploaded:
        with st.spinner("processing..."):
            path = './doc_files/'
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            
            process_document(path)
            st.rerun()

if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        role = message.get("role")
        content = message.get("content")
        st.chat_message(role).markdown(content)


    query = st.chat_input("Ask anything related to uploaded documents....")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})

        st.chat_message("user").markdown(query)
        response = st.session_state.agent.invoke(
            {"messages":[{"role":"user", "content":query}]},
            {"configurable":{"thread_id":1}}
        )

        answer = response["messages"][-1].content
        st.chat_message("ai").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})            


