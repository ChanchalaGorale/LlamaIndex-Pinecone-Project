from dotenv import load_dotenv
from pinecone import Pinecone
import os
from llama_index.core import VectorStoreIndex, ServiceContext
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.callback import LlamaDebugHandler, CallbackManager


load_dotenv()

pc = Pinecone(api_key=os.environ["PINECONE_API"])

index_name = "llama-docs"

if __name__=="__main__":

    pinecone_index = pc.Index(index_name)

    vector_store= PineconeVectorStore(pinecone_index=pinecone_index)

    llama_debug = LlamaDebugHandler(print_trace_on_end=True)

    callback_manager = CallbackManager([llama_debug])

    service_context = ServiceContext.from_defaults(callback_manager=callback_manager)

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)

    query= "What is llama index query engine?"

    query_engine= index.as_query_engine()

    response = query_engine(query)
    nodes = [(node, node.score) for node in response.source_nodes]




    

