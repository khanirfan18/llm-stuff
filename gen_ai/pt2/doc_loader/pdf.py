from langchain_community.document_loaders import PyMuPDFLoader


data = PyMuPDFLoader("pt2/rag_project/doc_loader/softc.pdf")

docs = data.load()

print(len(docs))