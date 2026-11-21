from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace,HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline

# embed using local model
embedding_model = HuggingFaceEmbeddings(
     model_name='sentence-transformers/all-MiniLM-L6-v2')

# local db - looking for the db
vectorstore = Chroma(
     persist_directory="rag_chroma_db",
     embedding_function=embedding_model)

# retriving from create_database.py

retriver = vectorstore.as_retriever(
     search_type="mmr",
     search_kwargs = {
          "k":4,
          "fetch_k":10,
          "lambda_mult":0.5
     })

################################################
# model setting locally
local_ai = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=800
)

# my_llm = HuggingFacePipeline.pipeline(
#      "text-generation",
#      model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#      max_new_tokens=800)

# need to create pipeline only for local models

my_llm = HuggingFacePipeline(pipeline=local_ai)

my_local_llm = ChatHuggingFace(llm=my_llm)
################################################

# prompt template

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a document question-answering assistant.

Follow these rules strictly:
1. Answer using ONLY the information explicitly available in the provided context.
2. Do not use external knowledge or make assumptions.
3. If the requested information is not present, respond exactly:
   I could not find the answer in the document.
4. When asked to list items, extract only the relevant items from the context.
5. Preserve the original names, numbering and wording.
6. Do not include headings, subheadings or unrelated information.
7. Never invent, modify or combine items.

Keep your answer concise."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])


print("Rag System Created ....")

print()

print('--------Press 0 to exit--------')

while True:
     query = input("You: ")
     if query == "0":
          break
     
     
     docs = retriver.invoke(query)
     
     context = "\n\n".join(
          [doc.page_content for doc in docs]
     )
     
     
     final_prompt = prompt.invoke({
          'context':context,
          'question':query
     })
     
     response = my_local_llm.invoke(final_prompt)
     
     
     print()
     
     
     print(f"\n AI: {response.content}")