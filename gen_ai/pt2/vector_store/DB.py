from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is a high-level programming language.",
        metadata={"topic": "Programming", "id": 1}
    ),
    Document(
        page_content="RAG combines retrieval systems with large language models.",
        metadata={"topic": "GenAI", "id": 2}
    ),
    Document(
        page_content="PostgreSQL is a relational database management system.",
        metadata={"topic": "Database", "id": 3}
    )
]

embedding_model = GoogleGenerativeAIEmbeddings(
     model="gemini-embedding-001"
)

# saves at a particular directory
vectorstore = Chroma.from_documents(
     documents= docs,
     embedding= embedding_model,
     persist_directory="chroma-db",
)

result = vectorstore.similarity_search("python is low level programming language?",k=2)

for r in result:
     print(r)
     
retriver = vectorstore.as_retriever()
docs = retriver.invoke()