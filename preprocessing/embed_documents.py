
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 1: Extract text from the PDF
doc = pymupdf.open("sample.pdf")
all_text = ""

for page in doc:
    all_text += page.get_text() + "\n"

doc.close()

# Step 2: Split the text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_text(all_text)

print("Total chunks:", len(chunks))

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert every chunk into an embedding
embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

# Step 5: Inspect the results
print("\nEmbedding shape:", embeddings.shape)

print("\nFirst chunk preview:")
print(chunks[0][:300])

print("\nFirst embedding (first 10 values):")
print(embeddings[0][:10])
