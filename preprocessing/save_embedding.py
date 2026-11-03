
import json
import numpy as np
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

# Step 4: Generate embeddings
embeddings = model.encode(
    chunks,
    show_progress_bar=True
)

print("Embedding shape:", embeddings.shape)

# Step 5: Save the text chunks
with open("chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

# Step 6: Save the embeddings
np.save("embeddings.npy", embeddings)

print("\nFiles saved successfully!")

# Step 7: Verify the saved files
with open("chunks.json", "r", encoding="utf-8") as file:
    saved_chunks = json.load(file)

saved_embeddings = np.load("embeddings.npy")

print("Saved chunks:", len(saved_chunks))
print("Saved embeddings shape:", saved_embeddings.shape)

print("First chunk:", saved_chunks[0][:100])
print("First embedding (first 5 values):", saved_embeddings[0][:5])

# Check that the chunks and embeddings are aligned
assert len(saved_chunks) == len(saved_embeddings)

print("\nVerification passed!")
