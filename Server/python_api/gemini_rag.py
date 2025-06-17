# project-root/Server/python_api/gemini_rag.py
from flask import Flask, request, jsonify
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from flask_cors import CORS
import logging # Import logging module
from dotenv import load_dotenv # Import load_dotenv

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv() # Load environment variables from .env file (should be in Server/python_api)

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# --- DIRECTORY PATHS ---
DOCUMENTS_DIR = "documents"
VECTORSTORE_DIR = "vectorstore"
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "index.faiss")
DOCUMENTS_LIST_PATH = os.path.join(VECTORSTORE_DIR, "documents.json")

# Ensure directories exist
os.makedirs(DOCUMENTS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# --- GLOBAL VARIABLES FOR RAG COMPONENTS ---
embedder = None
documents = [] # Stores the text chunks
faiss_index = None
# Initialize the Gemini model globally (recommended for performance)
# but configured with API key check for robustness
gemini_model = None

# --- FUNCTION TO CONFIGURE GEMINI API & MODEL ---
def configure_gemini():
    global gemini_model
    api_key = os.getenv("GEMINI_API_KEY") # Get API key from environment variable

    if not api_key:
        logging.error("CRITICAL ERROR: GEMINI_API_KEY environment variable not set. AI functionalities will not work.")
        return False # Indicate failure
    try:
        genai.configure(api_key=api_key)
        # Use the correct model name. 'gemini-1.5-flash-latest' is good.
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logging.info("Gemini API and model configured successfully.")
        return True # Indicate success
    except Exception as e:
        logging.error(f"Error configuring Gemini API or loading model: {e}")
        gemini_model = None # Ensure it's None on failure
        return False # Indicate failure

# --- FUNCTION TO LOAD OR CREATE VECTOR STORE ---
def load_or_create_vectorstore():
    global embedder, documents, faiss_index

    # Initialize embedder once (moved before the check for existing files)
    if embedder is None:
        logging.info("Loading SentenceTransformer model...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logging.info("SentenceTransformer model loaded.")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_LIST_PATH):
        logging.info("Loading existing FAISS index and documents...")
        try:
            faiss_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_LIST_PATH, "r", encoding="utf-8") as f:
                documents = json.load(f)
            logging.info(f"Loaded {len(documents)} document chunks and FAISS index from disk.")
        except Exception as e:
            logging.error(f"Error loading existing vector store: {e}. Rebuilding...")
            rebuild_vectorstore()
    else:
        logging.info("Vector store not found. Creating new vector store...")
        rebuild_vectorstore()

def rebuild_vectorstore():
    global documents, faiss_index

    # Path to the sample document
    sample_doc_path = os.path.join(DOCUMENTS_DIR, "sample.txt")

    # If sample.txt doesn't exist, create a dummy one for first run
    if not os.path.exists(sample_doc_path):
        logging.info(f"'{sample_doc_path}' not found. Creating a sample document for demonstration.")
        with open(sample_doc_path, "w", encoding="utf-8") as f:
            f.write("A contract is a legally binding agreement between two or more parties. For a contract to be valid and enforceable, several essential elements must generally be present. These include: Offer, Acceptance, Consideration, Mutual Assent, Legal Capacity, and Lawful Object. Contracts can be written, oral, or implied by conduct. However, some types of contracts, such as those involving real estate or those that cannot be performed within one year, may be required by law (Statute of Frauds) to be in writing to be enforceable. Breach of contract occurs when one party fails to fulfill their obligations as specified in the agreement, which can lead to remedies such as damages or specific performance.\n\n")
            f.write("In tort law, negligence is a legal theory under which a person can be held liable for injuries to another person caused by their failure to exercise reasonable care. To prove negligence, a plaintiff typically must establish four key elements: Duty of Care, Breach of Duty, Causation, and Damages. Defenses to negligence claims can include contributory negligence or assumption of risk.\n\n")
            f.write("Copyright law grants creators of original works of authorship exclusive rights to their works, such as books, music, and films. Protection arises automatically once an original work is fixed in a tangible medium. Exclusive rights include reproduction, distribution, performance, and display. Limitations like 'fair use' allow certain uses without permission. The duration of copyright typically lasts for the life of the author plus 70 years.\n\n")
            # Added some general knowledge examples to sample.txt, this makes the RAG
            # answer these specific general questions, rather than falling back.
            # If you want it to fallback for these, remove them from here.
            f.write("A car, or automobile, is a wheeled motor vehicle used for transportation.\n\n")
            f.write("Retrieval-Augmented Generation (RAG) is an AI framework that retrieves facts from an external knowledge base to ground large language models (LLMs) on authoritative sources and prevent hallucination.\n\n")
            f.write("MongoDB is a popular NoSQL database that uses JSON-like documents with optional schemas.\n\n")
            f.write("React is a free and open-source front-end JavaScript library for building user interfaces based on components.\n\n")
            f.write("FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.\n\n")
        logging.info(f"Sample '{sample_doc_path}' created.")

    with open(sample_doc_path, "r", encoding="utf-8") as f:
        # Simple chunking: split by double newline. Adjust as needed for larger/more complex docs.
        raw_text = f.read()
        documents = [chunk.strip() for chunk in raw_text.split("\n\n") if chunk.strip()]

    if not documents:
        logging.warning("No documents loaded from 'sample.txt'. RAG system will not function as expected.")
        # Initialize an empty index if no documents to prevent errors
        # Use a default embedding dimension if no documents for initial setup.
        # This dimension (384 for "all-MiniLM-L6-v2") needs to be known or dynamically determined.
        # It's safer to ensure embedder is loaded first.
        faiss_index = faiss.IndexFlatL2(embedder.get_sentence_embedding_dimension())
    else:
        logging.info(f"Embedding {len(documents)} document chunks...")
        doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
        logging.info("Embeddings created. Building FAISS index...")
        faiss_index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        faiss_index.add(doc_embeddings)
        logging.info("FAISS index built.")

        # Save the FAISS index and the document chunks for persistence
        faiss.write_index(faiss_index, FAISS_INDEX_PATH)
        with open(DOCUMENTS_LIST_PATH, "w", encoding="utf-8") as f:
            json.dump(documents, f)
        logging.info(f"Vector store with {len(documents)} document chunks saved to '{VECTORSTORE_DIR}'.")

# --- INITIALIZATION CALLS ---
# Call these functions once when the script starts
if configure_gemini(): # Only proceed if Gemini is configured successfully
    load_or_create_vectorstore()
else:
    logging.error("Gemini API not configured. RAG and direct AI calls will fail.")


# --- RETRIEVAL FUNCTION ---
def retrieve_chunks(query, top_k=3):
    if embedder is None or faiss_index is None or not documents:
        logging.warning("RAG system not fully initialized. Cannot retrieve chunks.")
        return []

    try:
        q_emb = embedder.encode([query], convert_to_numpy=True)
        # Ensure q_emb is 2D array for FAISS search (already handled by convert_to_numpy=True)

        # Check if query embedding dimension matches index dimension
        if q_emb.shape[1] != faiss_index.d:
            logging.error(f"Error: Query embedding dimension ({q_emb.shape[1]}) does not match index dimension ({faiss_index.d}). Rebuilding index might be necessary if model changed.")
            return []

        # Ensure top_k does not exceed the number of available documents
        actual_top_k = min(top_k, len(documents))
        if actual_top_k == 0:
            logging.info("No documents to retrieve from.")
            return [] # No documents to retrieve from

        _, indices = faiss_index.search(q_emb, actual_top_k)
        retrieved_chunks = [documents[i] for i in indices[0]]
        logging.info(f"Retrieved {len(retrieved_chunks)} chunks for query.")
        return retrieved_chunks
    except Exception as e:
        logging.error(f"Error during chunk retrieval: {e}", exc_info=True)
        return []

# --- API ROUTE ---
@app.route("/gemini-rag", methods=["POST"])
def rag_chat():
    global gemini_model

    prompt = request.json.get("prompt")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    logging.info(f"Received prompt: '{prompt}'")

    if gemini_model is None:
        return jsonify({"error": "AI service not initialized on the server."}), 503

    chunks = retrieve_chunks(prompt, top_k=3) # Retrieve top 3 relevant chunks

    final_prompt = prompt # Default to original prompt for fallback

    if chunks:
        context = "\n\n".join(chunks)
        # Explicitly instruct Gemini to use the context or state it can't answer
        final_prompt = f"Answer the following question only using the provided context. If the answer cannot be found in the context, state 'I don't have enough information to answer that based on the provided context.' Do not use external knowledge.\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
        logging.info("Using RAG context for prompt.")
    else:
        logging.info("No relevant chunks found. Falling back to Gemini's general knowledge.")
        # If no chunks, 'final_prompt' remains the original 'prompt',
        # allowing Gemini to use its general knowledge.

    try:
        logging.info(f"Sending prompt to Gemini. Final prompt length: {len(final_prompt)} characters.")
        response = gemini_model.generate_content(final_prompt)

        # Check if response.text exists and is not empty
        if hasattr(response, 'text') and response.text.strip():
            response_text = response.text.strip()
            logging.info(f"Gemini response: {response_text[:100]}...") # Log first 100 chars
            return jsonify({"response": response_text})
        else:
            logging.warning("Gemini API returned an empty or non-text response.")
            return jsonify({"error": "Gemini API returned an empty or non-text response. This might be due to safety filters or lack of relevant information."}), 500

    except genai.types.BlockedPromptException as e:
        logging.error(f"Gemini API blocked the prompt due to safety reasons: {e}")
        return jsonify({"error": f"VirLaw AI: Your prompt was blocked by AI safety features. Please rephrase."}), 400
    except genai.APIError as e:
        logging.error(f"Gemini API specific error: {e}", exc_info=True)
        return jsonify({"error": f"VirLaw AI: An API error occurred. Please try again. Details: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get response from Gemini: {str(e)}"}), 500

if __name__ == "__main__":
    # Ensure this is set correctly before running
    # If using .env, ensure the .env file is in the Server/python_api directory
    # os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE" # Not recommended to hardcode here

    # This call will now use the environment variable
    # configure_gemini() is called automatically at startup.
    app.run(host="0.0.0.0", port=8000, debug=True)