import json
import numpy as np

from sentence_transformers import SentenceTransformer, util

# Load the saved text chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load the saved embeddings
embeddings = np.load("embeddings.npy")

# print("Chunks loaded:", len(chunks))
# print("Embeddings shape:", embeddings.shape)

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# print("Embedding model loaded successfully!")

# Define a test question
test_questions = [
    {
        "question": "What is a trimmed mean, and how is it calculated?",
        "expected_chunks": [71, 72]
    },
    {
        "question": "What is the mean, and how is it calculated?",
        "expected_chunks": [70]
    },
    {
        "question": "What is the difference between mean and median?",
        "expected_chunks": [70, 74]
    },
    {
        "question": "What is a weighted mean, and how is it calculated?",
        "expected_chunks": [72]
    }
]

# Number of chunks to retrieve
top_k = 5

# Store Recall@5 scores
recall_scores = []

# Evaluate each question separately
for test_case in test_questions:

    question = test_case["question"]
    expected_chunks = test_case["expected_chunks"]

    print("\n" + "=" * 60)
    print("Question:", question)
    print("=" * 60)

    # Convert the current question into an embedding
    query_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Calculate similarity with all document embeddings
    similarity_scores = util.cos_sim(
        query_embedding,
        embeddings
    )[0]

    # Retrieve the top 5 chunks
    top_results = similarity_scores.topk(top_k)
    retrieved_chunks = top_results.indices.tolist()

    print("\nTop 5 retrieved chunks:")

    for index, score in zip(
        top_results.indices,
        top_results.values
    ):
        print(
            f"Chunk index: {index.item()}, "
            f"Similarity: {score.item():.4f}"
        )

    # Count relevant chunks retrieved
    relevant_retrieved = sum(
        1 for index in retrieved_chunks
        if index in expected_chunks
    )

    # Calculate Recall@5
    recall_at_5 = relevant_retrieved / len(expected_chunks)
    recall_scores.append(recall_at_5)

    print("\nExpected relevant chunks:", expected_chunks)
    print("Relevant chunks retrieved:", relevant_retrieved)
    print(f"Recall@5: {recall_at_5:.2f}")

# Calculate average Recall@5
average_recall = sum(recall_scores) / len(recall_scores)

print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

for i, score in enumerate(recall_scores, start=1):
    print(f"Question {i} Recall@5: {score:.2f}")

print(f"\nAverage Recall@5: {average_recall:.2f}")
print(f"Average Recall@5 percentage: {average_recall * 100:.1f}%")
