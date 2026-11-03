
import json
import numpy as np
import torch
import ollama

from sentence_transformers import SentenceTransformer, util

# Step 1: Load saved document chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Step 2: Load saved document embeddings
embeddings = np.load("embeddings.npy")
document_embeddings = torch.from_numpy(embeddings)

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Document chunks and embeddings loaded!")


# Step 4: Keep accepting questions until the user exits
while True:
    query = input("\nAsk a question about your PDF (or type 'exit'): ")

    if query.lower().strip() == "exit":
        print("Closing RAG application. Goodbye!")
        break

    if not query.strip():
        print("Please enter a question.")
        continue

    # Step 5: Convert the question into an embedding
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    
    # Step 6: Retrieve the top 3 most relevant chunks
    top_k = 5

    similarity_scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    top_results = similarity_scores.topk(top_k)

    print(f"\nRetrieving the top {top_k} chunks...")

    
    # Step 7: Collect and display retrieved passages
    retrieved_chunks = []

    print("\n" + "=" * 60)
    print("RETRIEVAL RESULTS")
    print("=" * 60)

    for rank, (index, score) in enumerate(
        zip(top_results.indices, top_results.values),
        start=1
    ):
        # Convert the tensor index and score into Python values
        chunk_index = index.item()
        similarity = score.item()

        # Get the original text for this chunk
        chunk_text = chunks[chunk_index]
        retrieved_chunks.append(chunk_text)

        # Display the retrieval result
        print(f"\nResult {rank}")
        print(f"Chunk index: {chunk_index}")
        print(f"Cosine similarity: {similarity:.4f}")
        print("\nPassage preview:")
        print(chunk_text[:400])
        print("-" * 60)

    # Combine the retrieved passages for the LLM
    context = "\n\n".join(retrieved_chunks)


    # Step 8: Build the prompt
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{query}

Instructions:
1. Use only information supported by the context.
2. Do not present assumptions or inferences as established facts.
3. If the context supports only part of the answer,
   explain which information is available and what is missing.
4. Do not invent advantages or disadvantages.
5. Keep the answer clear and concise.

"""

    # Step 9: Generate the answer with Ollama
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # Step 10: Display the answer
    answer = response["message"]["content"]

    print("\nRAG Answer:\n")
    print(answer)

