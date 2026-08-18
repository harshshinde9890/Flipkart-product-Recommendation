from langchain_astradb import AstraDBVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from flipkart.data_converter import DataConverter
from flipkart.config import Config


class DataIngestor:

    def __init__(self):

        # Local Hugging Face embedding model
        self.embedding = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )

        # AstraDB Vector Store
        self.vstore = AstraDBVectorStore(
            embedding=self.embedding,
            collection_name="flipkart_database",
            api_endpoint=Config.ASTRA_DB_API_ENDPOINT,
            token=Config.ASTRA_DB_APPLICATION_TOKEN,
            namespace=Config.ASTRA_DB_KEYSPACE
        )

    def ingest(self, load_existing=True):

        if load_existing == True:
            return self.vstore

        # Convert CSV data into LangChain documents
        docs = DataConverter(
            "data/flipkart_product_review.csv"
        ).convert()

        # Add documents to AstraDB
        self.vstore.add_documents(docs)

        return self.vstore


if __name__ == "__main__":

    ingestor = DataIngestor()

    ingestor.ingest(load_existing=False)