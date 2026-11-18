from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI




class Movie(BaseModel):
     title: str
     release_year: Optional[int]
     genre: List[str]
     director: Optional[str]
     cast: list[str]
     rating: Optional[float]
     summary : str
     

parser = PydanticOutputParser(pydantic_object=Movie)


llm = ChatGoogleGenerativeAI(
     model = 'gemini-3.5-flash-lite'
)

prompt = ChatPromptTemplate.from_messages([
      ('system',"""
       Extract Movie information from the paragraph
       {format_instructions}
       """),
      ('human',"""
       {paragraph}
       """)
      ])


paragraph_userinput = input("Give Your Para: ")


final_prompt = prompt.invoke(
     {
          "paragraph" : paragraph_userinput,
          "format_instructions": parser.get_format_instructions()
     }
)


response = llm.invoke(final_prompt)
print(response.text)
output = parser.parse(response.text)


print(output)



