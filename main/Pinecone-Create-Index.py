import os
import time
from pinecone import Pinecone, ServerlessSpec 
    
# Set your API keys for Pinecone
pc = Pinecone(
    api_key=os.environ['PINECONE_API_KEY']
)

# Create Index if not already created
pinecone_index_name = "langchain-embeddings-demo"
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name, 
        dimension=1536, # '1536' is the dimension for ada-002 embeddings
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
     
    while not pc.describe_index(pinecone_index_name).index.status['ready']:
        time.sleep(1)
    
    print("Pinecone Index provisioned")
else:
    print("Pinecone Index Already Provisioned")