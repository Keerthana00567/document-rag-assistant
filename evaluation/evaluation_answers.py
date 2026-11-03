
import json
import numpy as np
import torch
import ollama
from collections import Counter
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_recall_fscore_support


def evaluate_answer(reference_answer, generated_answer):

    # Convert answers into lowercase tokens
    reference_tokens = reference_answer.lower().split()
    generated_tokens = generated_answer.lower().split()

    # Count token occurrences
    reference_count = Counter(reference_tokens)
    generated_count = Counter(generated_tokens)

    # Count overlapping tokens, considering repetitions
    common_tokens = reference_count & generated_count

    matched_tokens = sum(common_tokens.values())

    # Calculate precision
    precision = matched_tokens / len(generated_tokens)

    # Calculate recall
    recall = matched_tokens / len(reference_tokens)

    # Calculate F1-score
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = (
            2 * precision * recall /
            (precision + recall)
        )

    return precision, recall, f1_score


def evaluate_faithfulness(context, generated_answer):

    evaluation_prompt = f"""
You are evaluating a RAG system.

Your task is to determine whether the generated answer
is supported by the retrieved context.

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Instructions:
1. Identify the factual claims in the generated answer.
2. Check whether each claim is supported by the context.
3. Identify unsupported claims.
4. Do not use outside knowledge.

Return your evaluation in this format:

Supported claims:
- List supported claims.

Unsupported claims:
- List unsupported claims.

Verdict:
Faithful or Not Faithful.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": evaluation_prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]


# Load document chunks
with open("chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Load saved document embeddings
embeddings = np.load("embeddings.npy")
document_embeddings = torch.from_numpy(embeddings)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("RAG evaluation resources loaded!")


test_cases = [
    {
        "question": "What is the mean, and how is it calculated?",
        "reference_answer": (
            "The mean is the sum of all values divided "
            "by the number of values."
        )
    },
    {
        "question": "What is a trimmed mean, and how is it calculated?",
        "reference_answer": (
            "A trimmed mean is calculated by removing a fixed "
            "number of the smallest and largest sorted values "
            "and then averaging the remaining values."
        )
    }
]

print("Number of test questions:", len(test_cases))


# Retrieve relevant chunks for each test question
top_k = 5

all_precisions = []
all_recalls = []
all_f1_scores = []
faithfulness_results = []

for test_case in test_cases:
    question = test_case["question"]

    # Convert question into an embedding
    query_embedding = model.encode(
        question,
        convert_to_tensor=True
    )

    # Calculate similarity with document chunks
    similarity_scores = util.cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    # Get the top 5 most relevant chunks
    top_results = similarity_scores.topk(top_k)

    # Collect the retrieved text
    retrieved_chunks = []

    for index in top_results.indices:
        chunk_index = index.item()
        retrieved_chunks.append(chunks[chunk_index])

    # Combine retrieved chunks into one context
    context = "\n\n".join(retrieved_chunks)

    print("\n" + "=" * 60)
    print("Question:", question)
    print("Retrieved chunks:", top_results.indices.tolist())
    print("Context length:", len(context), "characters")

    # Build the prompt using retrieved context
    prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Instructions:
1. Use only information supported by the context.
2. Do not invent facts or formulas.
3. If the context does not contain enough information,
   clearly say what is missing.
4. Keep the answer clear and concise.
"""

    # Generate the answer using Ollama

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    # Extract and display the generated answer
    generated_answer = response["message"]["content"]

    print("\nReference answer:")
    print(test_case["reference_answer"])

    print("\nGenerated RAG answer:")
    print(generated_answer)

    # Evaluate the generated answer

    # Evaluate the generated answer
    precision, recall, f1_score = evaluate_answer(
        test_case["reference_answer"],
        generated_answer
    )
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1_scores.append(f1_score)

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1_score:.2f}")

    print("-" * 60)

    # Evaluate answer faithfulness
    faithfulness_result = evaluate_faithfulness(
        context,
        generated_answer
    )
    faithfulness_results.append(faithfulness_result)
    print("\nFaithfulness Evaluation:")
    print(faithfulness_result)

    print("-" * 60)

# Calculate average evaluation metrics
average_precision = sum(all_precisions) / len(all_precisions)
average_recall = sum(all_recalls) / len(all_recalls)
average_f1 = sum(all_f1_scores) / len(all_f1_scores)

print("\n" + "=" * 60)
print("FINAL RAG EVALUATION RESULTS")
print("=" * 60)

print(f"Average Precision: {average_precision:.2f}")
print(f"Average Recall: {average_recall:.2f}")
print(f"Average F1-score: {average_f1:.2f}")

print("\nFaithfulness Results:")
for i, result in enumerate(faithfulness_results, start=1):
    print(f"\nQuestion {i}:")
    print(result)
