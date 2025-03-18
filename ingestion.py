# Import basics
import os
import time
from dotenv import load_dotenv

# Import Pinecone
from pinecone import Pinecone, ServerlessSpec

# Import LangChain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Import Documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Initialise pinecone database
index_name = os.environ.get("PINECONE_INDEX_NAME")

# Check if index exist, else create
existing_index = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_index:
    pc.create_index(
        name=index_name,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)

# Initialise embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-large",api_key=os.environ.get("OPENAI_API_KEY"))
# Initialise vector store
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Loading PDF documents
loader = PyPDFDirectoryLoader("documents/")
raw_documents = loader.load()

# Splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False
)

# Creating the chunks
documents = text_splitter.split_documents(raw_documents)

# Generate unique id
i = 0
uuids = []
while i < len(documents):
    i += 1
    uuids.append(f"id{i}")

# Add to vector database
vector_store.add_documents(documents=documents, ids=uuids)