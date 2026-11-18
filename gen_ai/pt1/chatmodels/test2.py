from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

my_llm = HuggingFaceEndpoint(
     repo_id="deepseek-ai/DeepSeek-V4.1-Flash-1017"
)

model = ChatHuggingFace(llm=my_llm)

response = model.invoke("Hey there, how are you?")

print(response.content)