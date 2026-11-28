from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


search_tool = TavilySearchResults(
     max_results = 5
)
# @tool
# def greet(name:str) -> str:
#      """
#      Generate a greeting message for a user
#      """
#      return f"Hello {name}, welcome to agentic things"

# greet_msg = greet.invoke('irfan')


print(search_tool.name)
print()
print()
print(search_tool.description)
print()
print()
print(search_tool.args)