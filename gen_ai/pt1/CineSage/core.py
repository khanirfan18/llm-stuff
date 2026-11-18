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
     ("system",
      """
      You are a movie information analyst.

     - Identify the movie title and release year.
     - Explain the central storyline in a concise summary.
     - Identify the main characters and their roles.
     - Mention the actors and director, if available.
     - Identify the movie's genre.
     - Highlight important themes and concepts explored.
     - Explain what makes the movie interesting or unique.
     - Mention any important relationships or conflicts in the story.

     Guidelines:
     - Only use information provided in the passage.
     - Do not invent or assume missing details.
     - Avoid unnecessary repetition.
     - Prioritize meaningful information over minor details.
     - Write in clear, simple, informative language.
     - If something is not mentioned, simply skip it.

     Present the extracted information using clear headings
     and bullet points.
     """),
     (
          "human",

          """      
          Analyze the following passage and extract the most useful
          and relevant information about the movie.
          Your tasks: {paragraph}
          """
     )
      
      ])


paragraph_userinput = input("Give Your Para: ")
final_prompt = prompt.invoke(
     {
          "paragraph" : paragraph_userinput
     }
)

response = llm.invoke(final_prompt)

print(response.text)