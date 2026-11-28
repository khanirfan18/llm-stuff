from langchain_community.document_loaders import WebBaseLoader

url = "https://www.graphify.com/"

data = WebBaseLoader(url)
result = data.load()

print(result[0].page_content)