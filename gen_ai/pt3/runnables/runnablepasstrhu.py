from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough




llm_model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.3
)


parser = StrOutputParser()


code_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a code generator"),
     ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a helpfull assistant who explains code in simple terms"),
     ("human", "Explain the following code in simple words:\n{code}")
])


seq = code_prompt | llm_model | parser 
seq2 = RunnableParallel(
     {
          "code": RunnablePassthrough(),
          "explanation" : explain_prompt | llm_model | parser
     }
)

chain = seq | seq2


result = chain.invoke({"topic":"write a code two sum"})


print(result['code'])
print()
print()
print(result['explanation'])