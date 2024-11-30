import os
import openai
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Set your API keys for OpenAI and Pinecone
openai.api_key = os.environ['OPENAI_API_KEY']

# Initialize OpenAI Embeddings using LangChain
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

# Connect to the Pinecone index using LangChain's Pinecone wrapper
pinecone_index_name = "langchain-embeddings-demo"
vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

# Define the retrieval mechanism
retriever = vector_store.as_retriever(search_kwargs={"k": 1})  # Retrieve top-3 relevant documents

# Initialize GPT-4 with OpenAI
llm = ChatOpenAI( model="gpt-4", openai_api_key=openai.api_key, temperature=0.7 )

# Define a Prompt Template for RetrievalQA
prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
)

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # Use the "stuff" method to combine documents into the prompt
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Example Query
query = "What are the forcast for  Central Bay of Campeche tonight   Nov 29th, 2024?"
response = qa_chain(query)

# Output the response and sources
print("Answer:", response["result"])
print("\nSources:")
for doc in response["source_documents"]:
    print(f"- {doc.metadata.get('text', 'Unknown Source')}")
