
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

import os
import requests

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage
from tavily import TavilyClient
from rich import print

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
llm = ChatGroq(model="openai/gpt-oss-20b",)

# Map tool names to their respective functions
tools = {
     "get_weather":get_weather,
     "get_news":get_news
}

# Bind available tools to the LLM
llm_with_tool = llm.bind_tools([get_weather,get_news])

# Agent LOOP: handles user input, tool calls and final responses

messages = []

print("City intelligence system")
print("type exit to quit")

while True:
     user_input = input("You : ")
     
     if user_input.lower() == "exit":
          break
     
     # Add user message to conversation history
     messages.append(HumanMessage(content = user_input))
     
     while True:
          # Let LLM decide whether to answer or call a tool
          result = llm_with_tool.invoke(messages)
          
          # Preserve LLM response in conversation history
          messages.append(result)
          
          # Check if the LLM requested any tool calls
          if result.tool_calls:
               for tool_call in result.tool_calls:
                    tool_name = tool_call['name']
                    
                    # Human in the loop: ask permission before execution
                    confirm = input(f"Agent wants to call {tool_name} Approve (y/n)")

                    if confirm.lower() == "n":
                         print("tool call denied!")
                         break
                    
                    # Execute the selected tool
                    
                    tool_result = tools[tool_name].invoke(tool_call)          
                              
                    # Send tool output back to LLM
                    messages.append(ToolMessage(
                         content=tool_result,
                         tool_call_id = tool_call['id']
                    ))
                    
               continue
          else:
               # Display final response when no tool call is needed
               print(result.content)
               break
