import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader 
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from htmlTemplate import css, bot_template, user_template

def extract_text_from_pdf_files(pdf_docs):
    raw_text = ""
    for pdf_doc in pdf_docs:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            raw_text += page.extract_text()
    return raw_text

def extract_chunks_from_text(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size = 1000, chunk_overlap = 200, length_function = len)
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_embeddings_from_text_chunks(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts = chunks, embedding = embeddings)
    return vector_store

def create_conversation_chain(vector_store):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_store.as_retriever(),
        memory = memory
    )
    return conversation_chain
    
def process_user_question(user_question):
    response = st.session_state.conversation_chain({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs")
    st.write("This is a simple web app to chat with PDFs. You can upload a PDF file and chat with it.")
    user_question = st.text_input("Ask a question about your PDF file:")

    if user_question:
        process_user_question(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your PDFs files")
        pdf_docs = st.file_uploader("Upload your PDF file", type=["pdf"], accept_multiple_files=True)
        
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = extract_text_from_pdf_files(pdf_docs)
                chunks = extract_chunks_from_text(raw_text)
                vector_store = get_embeddings_from_text_chunks(chunks)
                st.session_state.conversation_chain = create_conversation_chain(vector_store) 
                st.write("Processing Done!")   

if __name__ == "__main__":
    main()