from dotenv import load_dotenv
from langchain_core.messages import  AIMessage,SystemMessage,HumanMessage
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI


model = ChatGoogleGenerativeAI(
     model="gemini-3.5-flash-lite",
     temperature = 0.8,
     max_tokens = 1000
)

print("----Select Your Bot Mood----")
print("Press 1 For Angry Mood")
print("Press 2 For Sad Mood")
print("Print 3 For Funny Mood")

user_input = int(input())
if user_input == 1:
     mood = "You are an very very aggresive ai agent reply accordingly!"
elif user_input == 2:
     mood = "You are an very very sad ai agent reply accordingly!"
else:
     mood = "You are an very very funny ai agent reply accordingly!"

message = [
     SystemMessage(content=mood)
]

print('-------type 0 to exit the chatbot-------')
while True:
     prompt = input("You : ")
     message.append(HumanMessage(content=prompt))
     if prompt == "0":
          break
     else:
          response = model.invoke(message)
          message.append(AIMessage(response.content))

          print(f'Bot: {response.text}')


print(message)