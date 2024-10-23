from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import DirectoryLoader,TextLoader
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
## Langmith tracking
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

# load txt document to the vector DB
loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
documents = loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=250,chunk_overlap=200)
docs=text_splitter.split_documents(documents)

# load pdf to the vector DB
# from langchain_community.document_loaders import PyPDFLoader
# loader=PyPDFLoader('Doc/2023MSAPAccessPricing.pdf')
# docs=loader.load()
# text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
# documents=text_splitter.split_documents(docs)

# # PGVector needs the connection string to the database.
CONNECTION_STRING = "postgresql+psycopg://langchain:langchain@localhost:5433/langchain"
COLLECTION_NAME="vectordb"

# store the doc in the vector DB
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
store = PGVector.from_documents(
    docs,
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    use_jsonb=True,
)
