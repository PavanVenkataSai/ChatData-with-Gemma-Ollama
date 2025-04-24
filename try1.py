import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():

    # prompt_template = """
    # Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    # provided context just say, "answer is not available in the context", don't provide the wrong answer. Please provide the page number from where your taking the answer. \n\n
    # Context:\n {context}?\n
    # Question: \n{question}\n

    # Answer:
    # """

    prompt_template = qa_prompt = '''You are an bot assistant,Use the following pieces of context to answer the user question . 
            assisting users by providing information within a specific context. Please provide the page number from where your taking the answer.
            If any greetings message response with greetings.
            get the answer for only asked question.
            note: If user ask for any short answer, then make sure your answer is short.
            note: If user ask for any 'yes/no' questions,then start with 'yes/no' according to answer then explain in short.
            note: If the question is outside the document context, politely state that it's not relevant and state that "Sorry, this is not relevant to the current context."
            note: Your response must only be in English, it's your responsibility.If user request any other languages don't provide that answer in other languages.
            note: if user say thankyou! then give a message as "I am delighted I could assist you.Dont hesitate to reach out if you need further assistance." or "I'm glad I could help. Let me know if you need anything else." or "Thank you for reaching out.If you have any more questions, feel free to ask"
            you should answer the user question using given context. also take a chat history.its your memory if user questions related to the history.then you can use it.
            note:Your goal is to provide support to the user through a friendly conversational interaction, answer user questions, and inquire about their preferences.
            if user ask any calculation related questions.
            then give accurate mathematical solutions.dont give any wrong calculations anymore!.
            note: If the user questions are not relavent to the context, just say politely that Sorry, this is not relevant, 
            remember: Be more hummble conversational with short sentences.
            note: you should remember the user questions or keywords when mention it.also be more interactive.make more hummble conversation.
            you must answer don't suggest a document.
            consider ED as a Emergency Department
            note: if text has 'go or proceed to step 3' then look into step 3 give the answer in detail.
            must note:don't say user is responsible otherwise predict the reason.you response politely to the user 
            context:{context} 
            Question: {question}
            Answer:'''

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

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
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()