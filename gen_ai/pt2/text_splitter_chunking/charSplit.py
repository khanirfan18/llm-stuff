from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

data = TextLoader("gen_ai/pt2/rag_project/doc_loader/softc.txt")
splitter = CharacterTextSplitter(
     chunk_size = 10,
     chunk_overlap = 1,
     separator= ""
)
docs = data.load()
chunks = splitter.split_documents(docs)

for i in chunks:
     print(i.page_content)
     print()
     print()
     print()
