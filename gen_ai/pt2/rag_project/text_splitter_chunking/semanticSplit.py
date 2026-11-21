from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter,TokenTextSplitter
from google.genai import local_tokenizer



data = TextLoader("gen_ai/pt2/rag_project/text_splitter_chunking/ArtOfWar.pdf")
splitter = TokenTextSplitter(
     chunk_size = 1000,
     chunk_overlap = 10,
)


docs = data.load()
chunks = splitter.split_documents(docs)




print(chunks)
