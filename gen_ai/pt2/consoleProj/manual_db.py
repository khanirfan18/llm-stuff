from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()

my_doc = PyMuPDFLoader(
    'gen_ai/pt2/rag_project/consoleManual/SCPH-90002CB_90003CB_PS2_UG_EN.pdf')

doc = my_doc.load()

doc_chunks = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200).split_documents(doc)

hf_embed_model = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-001')

vector_store_db = Chroma.from_documents(
    documents=doc_chunks,
    embedding=hf_embed_model,
    persist_directory="chroma_console_db")
