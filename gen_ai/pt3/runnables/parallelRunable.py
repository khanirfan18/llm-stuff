from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda



# prompt template
short_prompt = ChatPromptTemplate.from_template(
     "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
     "Explain {topic} in detail"
)
# Model

llm_model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3
)

# output parser -> it structures the output properly
parser = StrOutputParser()
# input 
topic = "Machine Learning"

#############################################
######### - parallel Runnable - #############
######### lambda helps controlling input and custom logic
chain = RunnableParallel({
     "short":RunnableLambda(lambda x:x['short']) | short_prompt | llm_model | parser,
     "detailed":RunnableLambda(lambda x:x['detailed']) | detailed_prompt | llm_model | parser
})
#############################################

result = chain.invoke({
     "short" :{"topic":"Machine Learning"},
     "detailed":{"topic":"rise of soviet union"}
})

print('Short result ->>')
print()
print(result['short'])


print()


print('Long result ->>')
print()
print(result['detailed'])

