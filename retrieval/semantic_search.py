
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Step 1: Load the saved text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Step 2: Load the saved embeddings
embeddings = np.load("embeddings.npy")

print("Loaded chunks:", len(chunks))
print("Loaded embeddings:", embeddings.shape)

# Step 3: Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Enter a question
query = input("\nEnter your question:")

# Step 5: Convert the question into an embedding
query_embedding = model.encode(query, convert_to_tensor=True)

# Step 6: Convert saved embeddings into a tensor
document_embeddings = torch.from_numpy(embeddings)

# Step 7: Calculate cosine similarity
similarity_scores = util.cos_sim(
    query_embedding,
    document_embeddings
)[0]

# Step 8: Find the top 3 most similar chunks
top_k = 3
top_results = similarity_scores.topk(top_k)

# Step 9: Display the retrieved chunks
print("\nTop matching passages:\n")

for rank, (index, score) in enumerate(
    zip(top_results.indices, top_results.values), start=1
):
    chunk_index = index.item()
    similarity = score.item()

    print(f"Result {rank}")
    print(f"Chunk index: {chunk_index}")
    print(f"Cosine similarity: {similarity:.4f}")
    print("Text:")
    print(chunks[chunk_index])
    print("-" * 60)
