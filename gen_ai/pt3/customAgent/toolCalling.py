
from dotenv import load_dotenv
load_dotenv()

from langchain_groq.chat_models import ChatGroq
from langchain.tools import tool

from rich import print

#creating a tool

@tool
def get_text_len(text:str) -> int:
     """Returns the number of character in a given text"""
     return len(text)

llm = ChatGroq(model="openai/gpt-oss-20b")


#tool binding
# provide the available tools, eg: get_text_len
llm_with_tool = llm.bind_tools([get_text_len])


result = llm_with_tool.invoke("return the number of character in a given text: 'hello how are u my sonion'")
print(result.tool_calls)

# print(result)
# print()
# print('with tool below ->')
# print()
# print(result)
# tool_call = ""
if result.tool_calls:
     tool_call = result.tool_calls[0]
     tool_result = get_text_len.invoke(tool_call['args'])


tool_name = tool_call['name']
tool_args = tool_call['args']

tool_result = get_text_len.invoke(tool_args)

final_response = llm.invoke(f'the length of text is {tool_result}')


print(get_text_len.invoke(final_response.content[0]))
# print(final_response.content)


# final_res = llm.invoke(
#      f"the len of the text is {tool_result}"
# )
# print(final_res.content)
