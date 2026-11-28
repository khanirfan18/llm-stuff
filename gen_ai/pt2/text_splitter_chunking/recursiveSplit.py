from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,TokenTextSplitter,RecursiveCharacterTextSplitter
from google.genai import local_tokenizer
from langchain_community.document_loaders import PyMuPDFLoader


data = PyMuPDFLoader("gen_ai/pt2/rag_project/text_splitter_chunking/ArtOfWar.pdf")
splitter = RecursiveCharacterTextSplitter(
     chunk_size = 1000,
     chunk_overlap = 200,
)


docs = data.load()
chunks = splitter.split_documents(docs)

print('this is length of chunks :',len(chunks))
print('this is length of docs :',len(docs))

