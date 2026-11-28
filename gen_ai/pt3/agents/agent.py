
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

import os
import requests

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print
from langchain.agents import create_agent

# Middleware not applied yet! #


# OpenWeather tool: fetch current weather using city name
@tool
def get_weather(city:str) -> str:
     """Get current weather of a city"""
     WEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
     url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric"
     
     response = requests.get(url)
     data = response.json()
     
     print(f'the data :- {data}')
     
     # Check API response before extracting weather details
     if response.status_code != 200:
          return f"Error - Something went wrong!"
     
     # Extract required fields from JSON response
     temp = data["main"]["temp"]
     desc = data["weather"][0]["description"]
     
     return f"Weather in {city}: {desc}, {temp} 'C "




# Tavily News tool: search for recent news about a city
tavily_client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))


@tool
def get_news(city:str)-> str:
     """Get latest city news"""
     
     # Build search query dynamically
     query = f"latest news in {city}"
     
     response = tavily_client.search(
          query = query,
          search_depth="basic",
          max_results=3
     )
     
     # Extract search results from response
     result = response.get("results",[])
     
     if not result:
          return f"No news found for {city}"
     
     # Format search results into a readable list
     news_list = []
     
     for r in result:
          title = r.get("title","No title")
          url = r.get("url","")
          snippet = r.get("content","")
          
          news_list.append(f"-{title}\n {url}\n {snippet[:100]}...")
          
          
     return f"latest news in {city}:\n\n" + "\n\n".join(news_list)


# Initialize LLM
llm = ChatGroq(model="openai/gpt-oss-20b")

agent = create_agent(
     llm,
     tools=[get_news,get_weather],
     system_prompt = "You are a helpfull city assistant!"
)

print("City Agent")

while True:
     user_input = input("You : ")
     if user_input.lower() == "exit":
          break
     result = agent.invoke({
          "messages" : [{
               "role":"user",
               "content":user_input
          }]
     })
     
     print("Bot : ", result)
