from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings



from langchain_core.documents import Document

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used to minimize a loss function by iteratively updating model parameters in the direction opposite to the gradient."),
    
    Document(page_content="Learning rate controls the size of each parameter update during gradient descent. A very high learning rate can cause overshooting, while a very low learning rate makes training slow."),
    
    Document(page_content="In machine learning, the cost function measures how far a model's predictions are from the actual values. Gradient descent attempts to find parameters that minimize this cost."),
    
    Document(page_content="Stochastic Gradient Descent (SGD) updates model parameters using the gradient calculated from a single randomly selected training example instead of the entire dataset."),
    
    Document(page_content="Batch Gradient Descent calculates the gradient using the entire training dataset before updating model parameters. It provides stable updates but can be computationally expensive.")
]

hf_embed_model = HuggingFaceEmbeddings(
     model_name='sentence-transformers/all-MiniLM-L6-v2')


# saves it in ram
vectorstore = Chroma.from_documents(docs, hf_embed_model)

similarity_retriever = vectorstore.as_retriever(
     search_type="similarity",
     search_kwargs={"k":3}, # give 3 top k results/output
     
)

print("----here is your similarity results----")

similarity_docs = similarity_retriever.invoke("what is gradient ")

for doc in similarity_docs:
     print()
     print("----")
     print(doc.page_content)
     print("---")
     print()