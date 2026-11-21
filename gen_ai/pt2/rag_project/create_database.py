#load pdf 
#split into chunks
#create the embeddings
#store into chroma
from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

data = PyMuPDFLoader("gen_ai/pt2/rag_project/ArtOfWar.pdf")
docs = data.load()

#########################################
# chunking the doc content
splitter = RecursiveCharacterTextSplitter(
     chunk_size = 1000,
     chunk_overlap = 200
)

chunks = splitter.split_documents(docs)
#########################################



#########################################
# Embed Model :
hf_embed_model = HuggingFaceEmbeddings(
     model_name='sentence-transformers/all-MiniLM-L6-v2')
#########################################



#########################################
# Local Database
vectorstore = Chroma.from_documents(
     documents= chunks,
     embedding= hf_embed_model,
     persist_directory= "rag_chroma_db")

#########################################




# Testing the db...
result = vectorstore.similarity_search(
    "list all 13 chapters of Sun Tzu's Art of War",
    k=4
)

for r in result:
    print(r.page_content)
    print(r.metadata)
