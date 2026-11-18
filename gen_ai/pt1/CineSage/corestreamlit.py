
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: list[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=Movie)


llm = ChatGoogleGenerativeAI(
    model='gemini-3.5-flash-lite'
)


prompt = ChatPromptTemplate.from_messages([
    ('system', """
        Extract Movie information from the paragraph
        {format_instructions}
    """),
    ('human', """
        {paragraph}
    """)
])


# Streamlit UI

st.set_page_config(
    page_title="CineSage",
    page_icon="🎬",
    layout="centered"
)

st.title("CineSage")
st.caption("AI-powered movie information extractor")

st.divider()

paragraph_userinput = st.text_area(
    "Movie passage",
    placeholder="Paste your movie description here...",
    height=180
)

if st.button("Extract Movie Information", use_container_width=True):

    if paragraph_userinput.strip():

        try:
            with st.spinner("Analyzing movie..."):

                final_prompt = prompt.invoke({
                    "paragraph": paragraph_userinput,
                    "format_instructions": parser.get_format_instructions()
                })

                response = llm.invoke(final_prompt)

                output = parser.parse(response.text)

            st.divider()

            st.subheader(output.title)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Release Year",
                    output.release_year if output.release_year is not None else "N/A"
                )

            with col2:
                st.metric(
                    "Rating",
                    output.rating if output.rating is not None else "N/A"
                )

            st.write("**Director**")
            st.write(output.director or "N/A")

            st.write("**Genre**")
            st.write(", ".join(output.genre) if output.genre else "N/A")

            st.write("**Cast**")
            st.write(", ".join(output.cast) if output.cast else "N/A")

            st.write("**Plot Summary**")
            st.write(output.summary)

            st.success("Extraction completed!")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

    else:
        st.warning("Please enter a movie passage.")

