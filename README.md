# video_001502
# 
Contents:
This example python langchain codebase shows an example of how to use RAG when invoking an LLM.  The Embeddings used in the RAG workflow are queried from a Pinecone Vector Database.  The OpenAI gpt LLM is used in this example. I also included some helper python scripts to create/delete the Pinecone index we use.

Helper scripts:
Pinecone-Create-Index.py - This script creates an index in the Pinecone vector database. The servless Pinecone vector database service is in AWS cloud US-East-1 region.
Pinecone-Delete-Index.py - This script deletes the index, we created, in the Pinecone vector database.

Embedding Scripts:
LangChain-Embedding-From-Dir.py - This script, will create embeddings, using all of the text files in a directory, an insert these embeddings into the Pinecone index we created.

RAG Scripts:
LangChain-RAG-Example.py - This script, executes a RAG workflow. It retrieves Embeddings from a Pinecone vector database.  It pairs up the Embeddings with the original user query to invoke and OpenAI gpt LLM.

3rd Party Services:
You need to get an account and API key from OpenAI and you need an account and API key for Pinecone.

Run Sequence:
If you run this the first time, and you don't anything in place, you can do the following:
1. Run Pinecone-Create-Index.py
2. Run LangChain-Embedding-From-Dir.py
3. Login Pinecone, view your index and Embeddings inserted into them.
4. Run LangChain-RAG-Example.py

Important Notes:
Index Dimension - When you select your Embedding model, determine what size (or dimension) of the vectors it will generate. When you create your Pinecone index, you should specify the same dimension setting to match your Embedding model.
Environment Variables - I have set my OPENAI_KEY and PINECONE_KEY as environment variables.  The Python scripts read these environment variables when they need these values.  See instructions depending your OS as its a little different for Window OS versus Mac OS.

Dependencies:
pip3 install pinecone-client
pip3 install langchain
pip3 install langchain-core
pip3 install langchain-community
pip3 install langchain-openai
pip3 install langchain-pinecone
pip3 install langchain-text-splitters
