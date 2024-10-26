import streamlit as st 
from htmlTemplates import css, bot_template, user_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import HuggingFaceHubEmbeddings
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
import os
import re


DATA_DIR="C:\\Users\\utente\\OneDrive\\Bureau\\final_project\\data\\test"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = ""





def save_file(uploaded_file):
    """Save the uploaded file to the data directory."""
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    if not os.path.exists(file_path):
        # Write the file to the specified directory
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return uploaded_file.name
    return uploaded_file.name






def save_upload_file_temporary(uploaded_file):
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf", dir="C:\\Users\\utente\\OneDrive\\Bureau\\final_project\\data\\test") as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name  # Returns the file path of the saved file
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None






def get_pdf(directory):
    loader = DirectoryLoader(directory,show_progress=True)
    documents = loader.load()
    return documents






def get_pdf_chunks(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
  docs = text_splitter.split_documents(documents)
  return docs





def get_vectorstore(docs):
    persist_directory = 'db6'
    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key="", model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    #persist_directory=persist_directory
    )
    #vectordb.persist()
    return vectordb







def get_conversation_chain(vectordb):
    llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
    )
        
        
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.chains import ConversationalRetrievalChain

    memory = ConversationBufferWindowMemory(k=1)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                  chain_type="stuff",
                                  retriever = vectordb.as_retriever(search_kwargs={"k": 2}),
                                  memory=memory,
                                  return_source_documents=True)
    return chain







def handle_userinput(input):
    chat_history = []
    result = st.session_state.conversation({'question': input, 'chat_history': chat_history})
    #response = st.session_state.conversation({'query':input})
    chat_history.append((input, result['answer']))
    helpful_answer_pattern = r"Helpful Answer:\s*(.*)"
    # Using regex to find the helpful answer
    helpful_answer_match = re.search(helpful_answer_pattern, result['answer'])
    helpful_answer = helpful_answer_match.group(1) if helpful_answer_match else "No helpful answer found"
    st.write(helpful_answer)
    
  
  
  
  
  
  
  
def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        
        st.session_state.chat_history = None
      
    st.header("Chat with your resumes")
    user_question = st.text_input("Ask a question about your documents:")
    print(user_question)
    if user_question:
        handle_userinput(user_question)
        
        
    st.header("Ask your CSV 📈")
        
    # Set up the sidebar for uploading PDF files
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your files here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            
           if pdf_docs is not None and len(pdf_docs) > 0:
               
            with st.spinner("Processing"):
                
                for pdf_file in pdf_docs:
                    
                    
                    file_path = save_upload_file_temporary(pdf_file)
                    print(pdf_file)
                    st.success("leprocessing bdee")
                    if file_path:
                        raw_text = get_pdf(DATA_DIR)
                        os.unlink(file_path)
                        st.write("Processed text from uploaded PDF:")
                        st.text_area("Text", value=raw_text, height=300)
                        text_chunks = get_pdf_chunks(raw_text)
                        print(len(text_chunks))
                        vectorstore = get_vectorstore(text_chunks)
                        st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                        #os.unlink(file_path)  # Clean up the temporary file
                        #os.remove(file_path)
                    else:
                        st.error("Please upload at least one PDF file.")
            
                       
if __name__ == "__main__":
    main()
    
    