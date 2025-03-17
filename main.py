import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from google.colab import files
from langchain_community.embeddings import OpenAIEmbeddings

# Load API keys
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Configure Google Generative AI
genai.configure(api_key="AIzaSyBgJDRTkjwujipEvegv6Le7U9DeprzcPGg", transport="rest")
# PDF Extraction
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Text Chunking
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Vector Store Creation
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Question-Answering Chain
def get_conversational_chain(vector_store):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say: "Answer is not available in the context."
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chat_model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash-latest")


    return ConversationalRetrievalChain.from_llm(
        llm=chat_model,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Process User's Question
def user_input(question, chain, chat_history):
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    print("Reply:", response["answer"])
    chat_history.append((question, response["answer"]))

# Main Execution
print("📂 Upload your PDF files")
uploaded = files.upload()

pdf_docs = [pdf for pdf in uploaded.keys()]
raw_text = get_pdf_text(pdf_docs)
text_chunks = get_text_chunks(raw_text)
get_vector_store(text_chunks)
print("✅ PDF processed successfully!")

# Initialize Chain
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
faiss_index = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
chain = get_conversational_chain(faiss_index)

# Ask Questions
chat_history = []
while True:
    user_question = input("Ask a question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        print("👋 Session ended.")
        break
    user_input(user_question, chain, chat_history)
