### COMPLETE THE CODE ###

from policy_loader import load_policy_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

## TO LOAD THE DOCUMENT, USE THE FOLLOWING ONLY:
documents = load_policy_documents()

# Step 2: Split into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
splits = text_splitter.split_documents(documents)

# Step 3: Create embeddings and build FAISS vector store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(splits, embeddings)

# Step 4: Create retriever (top-3 most relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
