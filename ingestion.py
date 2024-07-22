import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import (
    download_loader,
    ServiceContext,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
import logging
import sys
from llama_index.readers.file import UnstructuredReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API"])

index_name = "llama-docs"

index = pc.Index(index_name)

if __name__ == "__main__":
    unstructuredreader = download_loader("UnstructuredReader")
    dir_reader = SimpleDirectoryReader(
        input_dir="./llamaindex_docs", file_extractor={".html": UnstructuredReader()}
    )
    documents = dir_reader.load_data()

    node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=20)

    # nodes = node_parser.get_nodes_from_documents(documents=documents)

    llm = OpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )
    embed_model = OpenAIEmbedding(mode="text-embedding-ada-002", embed_batch_size=100)

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model, node_parser=node_parser
    )

    vector_store = PineconeVectorStore(pinecone_index=index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        service_context=service_context,
        show_progress=True,
    )

    
