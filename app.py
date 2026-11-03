
import streamlit as st
import torch
import ollama
import fitz
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util


# =====================================================
# 1. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="centered"
)


# =====================================================
# 2. LOAD AND CACHE SENTENCE TRANSFORMER MODEL
# =====================================================

@st.cache_resource
def load_resources():

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


sentence_model = load_resources()


# =====================================================
# 3. GENERATE AND CACHE DOCUMENT EMBEDDINGS
# =====================================================

@st.cache_resource
def generate_embeddings(chunks, file_hash):

    embeddings = sentence_model.encode(
        chunks,
        convert_to_tensor=True,
        show_progress_bar=True
    )

    return embeddings


# =====================================================
# 4. APPLICATION HEADING
# =====================================================

st.title("📚 PDF RAG Assistant")

st.markdown(
    "Upload a document and interact with it using AI."
)

st.caption(
    "Powered by Sentence Transformers, PyTorch and Ollama"
)

st.divider()


# =====================================================
# 5. CACHED PDF EXTRACTION AND CHUNKING
# =====================================================

@st.cache_data
def process_pdf(pdf_bytes):

    pdf_document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    full_text = ""

    for page in pdf_document:
        full_text += page.get_text() + "\n"

    pdf_document.close()

    if not full_text.strip():
        return [], 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_text(full_text)

    return chunks, len(full_text)


# =====================================================
# 6. PDF UPLOAD
# =====================================================

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)


# =====================================================
# 7. INITIALIZE CHAT HISTORY
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =====================================================
# 8. DETECT DOCUMENT CHANGES
# =====================================================

if uploaded_file is not None:

    file_hash = hashlib.md5(
        uploaded_file.getvalue()
    ).hexdigest()

    if st.session_state.get("current_file_hash") != file_hash:

        st.session_state.messages = []

        st.session_state.current_file_hash = file_hash


# =====================================================
# 9. PROCESS UPLOADED PDF
# =====================================================

if uploaded_file is not None:

    # Extract and chunk PDF
    try:

        chunks, total_characters = process_pdf(
            uploaded_file.getvalue()
        )

    except Exception as e:

        st.error(f"Unable to process PDF: {e}")
        st.stop()

    # Check extracted text
    if not chunks:

        st.error(
            "No text could be extracted from this PDF. "
            "It may be a scanned document."
        )

        st.stop()

    st.success("PDF text extracted successfully!")

    # Generate document embeddings
    try:

        with st.spinner("Generating document embeddings..."):

            document_embeddings = generate_embeddings(
                chunks,
                file_hash
            )

        st.success(
            "PDF processed and embeddings generated!"
        )

    except Exception as e:

        st.error(f"Embedding generation failed: {e}")
        st.stop()

    # Display document information
    st.write("### Document successfully processed!")

    # Sidebar
    with st.sidebar:

        st.title("📚 Document Details")

        st.markdown("---")

        st.subheader("Uploaded PDF")

        st.write(uploaded_file.name)

        st.caption(
            f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )

        st.markdown("---")

        st.subheader("Processing Information")

        st.metric(
            "Total Characters",
            f"{total_characters:,}"
        )

        st.metric(
            "Total Chunks",
            len(chunks)
        )

        st.metric(
            "Embedding Dimensions",
            document_embeddings.shape[1]
        )

        st.markdown("---")

        st.success("Document Ready")

    # Clear chat button
    with st.sidebar:

        st.markdown("---")

        if st.button(
            "🗑️ Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()

    # =================================================
    # 10. CHAT INPUT
    # =================================================

    user_question = st.chat_input(
        "Ask a question about your PDF..."
    )

    # =================================================
    # 11. PROCESS USER QUESTION
    # =================================================

    if user_question:

        # Generate query embedding
        query_embedding = sentence_model.encode(
            user_question,
            convert_to_tensor=True
        )

        # Calculate cosine similarity
        similarities = util.cos_sim(
            query_embedding,
            document_embeddings
        )[0]

        # Retrieve top relevant chunks
        top_k = min(5, len(chunks))

        top_results = torch.topk(
            similarities,
            k=top_k
        )

        # Extract retrieved chunks
        retrieved_chunks = [
            chunks[idx]
            for idx in top_results.indices.tolist()
        ]

        # Extract similarity scores
        scores = top_results.values.tolist()

        # Construct context
        context = "\n\n".join(retrieved_chunks)

        # =============================================
        # 12. CREATE RAG PROMPT
        # =============================================

        prompt = f"""
        Answer the question using only the provided context.

        If the context does not contain the answer,
        clearly state that the information is not available
        in the document.

        Do not invent information.

        Context:
        {context}

        Question:
        {user_question}

        Answer:
        """

        # =============================================
        # 13. GENERATE ANSWER USING OLLAMA
        # =============================================

        try:

            with st.spinner(
                "Searching document and generating answer..."
            ):

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

                answer = response["message"]["content"]

        except ollama.ResponseError as e:

            st.error(f"Ollama error: {e}")
            st.stop()

        except Exception as e:

            st.error(f"Unable to generate answer: {e}")
            st.stop()

        # =============================================
        # 14. STORE USER QUESTION
        # =============================================

        st.session_state.messages.append({

            "role": "user",

            "content": user_question

        })

        # =============================================
        # 15. STORE ASSISTANT RESPONSE AND SOURCES
        # =============================================

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "sources": retrieved_chunks,

            "scores": scores

        })

    # =================================================
    # 16. DISPLAY CHAT HISTORY
    # =================================================

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            # Display retrieved sources
            if message["role"] == "assistant":

                with st.expander("View Retrieved Sources"):

                    for i, (chunk, score) in enumerate(
                        zip(
                            message["sources"],
                            message["scores"]
                        )
                    ):

                        st.markdown(
                            f"**Source {i + 1}**"
                        )

                        st.caption(
                            f"Similarity Score: {score:.4f}"
                        )

                        st.write(chunk)

                        st.divider()


# =====================================================
# 17. MESSAGE WHEN NO PDF IS UPLOADED
# =====================================================

else:

    st.info(
        "Please upload a PDF document to start chatting."
    )
