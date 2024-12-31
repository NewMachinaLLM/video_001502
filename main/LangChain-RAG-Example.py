from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate

import os
import openai

# Set your API keys for OpenAI and Pinecone
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI Embeddings using LangChain
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

# Connect to the Pinecone index using LangChain's Pinecone wrapper
pinecone_index_name = "langchain-embeddings-demo"
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

# Define the retrieval mechanism
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top-1 relevant documents

# Initialize GPT-4 with OpenAI
llm = ChatOpenAI( model="gpt-4", openai_api_key=openai.api_key, temperature=0.7 )

# Define Prompt Template
prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

# Create LLM Chain
llm_chain = prompt_template | llm | StrOutputParser()

# Retrieve documents
query = "What are the forcast for  Central Bay of Campeche tonight   Nov 29th, 2024?"
docs = retriever.invoke(query)
context = "\n\n".join([doc.page_content for doc in docs])
    
# Run LLM chain with the retrieved context
answer = llm_chain.invoke({"context": context, "question": query})

# Output the Answer and Sources
print("Answer:", answer)