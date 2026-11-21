from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate


# 1. Embedding model
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# 2. Load existing Chroma database
vectorstore = Chroma(
    persist_directory="chroma_console_db",
    embedding_function=embedding_model
)


# 3. Create retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)


# 4. Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


# 5. Create prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.
        Answer using ONLY the provided context.
        If the answer is not present, say:
        I could not find the answer to that question... ;-;
        """
    ),
    (
        "human",
        """
        Context: {context}

        Question: {question}
        """
    )
])


# 6. User question
user_question = "How do you turn off the console and put it into standby mode?"


# 7. Retrieve relevant documents
docs = retriever.invoke(user_question)


# 8. Combine retrieved chunks into context
context = "\n\n".join(
    doc.page_content for doc in docs
)


# 9. Format the prompt
final_prompt = prompt.invoke({
    "context": context,
    "question": user_question
})


# 10. Generate response
response = llm.invoke(final_prompt)


# 11. Display answer
print(f"\nAI: {response.content}")


for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(f" doc page content -> {doc.page_content}")
    
    
print(f"\nRetrieved chunks: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i+1} ---")
    print(doc.page_content)
    
print("Total documents in Chroma:", vectorstore._collection.count())