from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



# prompt template
prompt = ChatPromptTemplate.from_template(
     "Explain {topic} in simple words"
)

# Model

llm_model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3
)

# output parser -> it structures the output properly
parser = StrOutputParser()

# format the prompt
formatted_prompt = prompt.format_messages(topic='Machine Learning')

# Call the model manually
response = llm_model.invoke(formatted_prompt)

# parse the output manually
final_output = parser.parse(response.content)



# this skips parser initialization,parser usage,formatting 
#############################################
######### - Sequence Runnable - #############
chain = prompt | llm_model | parser 
result = chain.invoke("Machine Learning")

print(result)

#############################################