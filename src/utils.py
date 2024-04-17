from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)


def clean_elements(elements: list, **kwargs) -> None:
    for element in elements:
        element.text = clean(element.text, **kwargs)
    return elements


def chunk_text(filename: str) -> list:
    return clean_elements(
        partition(
            filename=filename,
            strategy="fast",
            chunking_strategy="basic",
            overlap_all=True,
            max_characters=500,
        ),
        extra_whitespace=True,
        bullets=True,
    )


# Linked list with chunks?
def load_db(elements: list) -> Chroma:
    documents = []
    for i, element in enumerate(elements):
        metadata = element.metadata.to_dict()
        del metadata["languages"]
        metadata["source"] = metadata["filename"]
        # elem_index is used to allow for additional context from surrounding text
        metadata["elem_index"] = i
        documents.append(Document(page_content=element.text, metadata=metadata))

    return Chroma.from_documents(
        documents, SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )
