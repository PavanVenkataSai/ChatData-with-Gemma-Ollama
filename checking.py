import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz
import tabula
from langchain_community.chat_models import ChatOllama

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    extracted_data = []
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        page_text = page.get_text("text")
        tables = tabula.read_pdf(pdf_path, pages=page_number + 1, multiple_tables=True, stream=True)

        if tables:
            table_data = []
            for table in tables:
                table_data.append(table.to_dict(orient="records"))
            extracted_data.append(page_text.strip())
            extracted_data.append(table_data)
        else:
            extracted_data.append(page_text.strip())

    pdf_document.close()
    text = ''.join([str(item) for item in extracted_data])

    return text

def get_text_chunks(text, max_chunk_size=2048):
    # Split text into smaller chunks based on max_chunk_size
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        if end >= len(text):
            end = len(text)
        chunks.append(text[start:end])
        start = end
    # print(f'length of chucks : {len(chunks)}')
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    prompt_template = """
    You are an expert in extracting information from technical manuals.
    You will be provided with document. Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # model = ChatGoogleGenerativeAI(model="gemini-pro",
    #                          temperature=0.3)

    model = ChatOllama(model="gemma:2b")

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    print(f'--->Quesiton : {user_question}\n\n\n\n--->Docs : {docs}')
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF File and Click on the Submit & Process Button", type=["pdf"])
        if st.button("Submit & Process") and pdf_doc is not None:
            with st.spinner("Processing..."):
                save_folder = "uploads"  # Specify the folder to save the uploaded files
                os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist
                file_path = os.path.join(save_folder, "uploaded_file.pdf")  # Construct the file path
                with open(file_path, "wb") as f:
                    f.write(pdf_doc.getbuffer())

                # filepath = 'C:\Users\CVHS\pavan\chat_gemma\langchain-gemma-ollama-chainlit\'
                print(file_path)
                
                raw_text = get_pdf_text(file_path)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()