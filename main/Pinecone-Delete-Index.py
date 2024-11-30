import os
from pinecone import Pinecone

# Set your API keys for Pinecone
pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY']
)

# Create Index if not already created
pinecone_index_name = "langchain-embeddings-demo"
if pinecone_index_name in pc.list_indexes().names():
    pc.delete_index( name=pinecone_index_name )
    
    print("Pinecone Index Deleted")
else:
    print("Pinecone Index Had Already been Deleted")