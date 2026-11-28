# instead of using one query we use mulitple query to 
# get better result by retriving

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
# we gotta import this specifically unlike (mmr,similarity)
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from transformers import pipeline

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

# my_local_llm = ChatHuggingFace(
#      model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#      task="text-generation",
#      pipeline_kwargs={"max_new_tokens": 800,}
# )

#############################################
####### ONLY For Local Models ###############


pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=800
)

llm = HuggingFacePipeline(pipeline=pipe)

my_local_llm = ChatHuggingFace(llm=llm)


#############################################

# saves it in ram
vectorstore = Chroma.from_documents(docs, hf_embed_model)


data_retriever = vectorstore.as_retriever()


multi_query_retriever = MultiQueryRetriever.from_llm(
     retriever=data_retriever,
     llm= my_local_llm # give 3 top k results/output
     
)

mult_query_docs = multi_query_retriever.invoke("what is gradient descent")

print("----here is your similarity results----")


for doc in mult_query_docs:
     print()
     print("----")
     print(doc.page_content)
     print("---")
     print()