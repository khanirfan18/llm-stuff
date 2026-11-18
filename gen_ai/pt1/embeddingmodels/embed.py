from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
     model = 'gemini-embedding-001',
     dimensions = 64
     )

vector = embeddings.embed_query("You are going to learn gen-ai")

print(vector)