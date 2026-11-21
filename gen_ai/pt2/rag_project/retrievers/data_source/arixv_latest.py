import arxiv
from langchain_core.documents import Document

client = arxiv.Client()
search = arxiv.Search(
    query="large language models",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance,
)

docs = [
    Document(
        page_content=result.summary,
        metadata={
            "Published": result.published.strftime("%Y-%m-%d"),
            "Title": result.title,
            "Authors": ", ".join(a.name for a in result.authors),
            "entry_id": result.entry_id,
        },
    )
    for result in client.results(search)
]

print(docs[0].metadata["Title"])