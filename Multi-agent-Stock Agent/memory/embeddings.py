from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

_embedding_instance = None


def get_embeddings():
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001"
        )
    return _embedding_instance