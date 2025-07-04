from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import tempfile
import traceback
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import shutil
import atexit
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration - Updated for 200MB file size
app.config.update({
    "TIMEOUT": 600,
    "MAX_CONTENT_LENGTH": 200 * 1024 * 1024,  # 200MB
    "UPLOAD_FOLDER": os.path.join(tempfile.gettempdir(), "pdf_chatbot_uploads"),
    "FEEDBACK_FILE": "user_feedback.json",
    "MODEL_WEIGHTS_FILE": "model_weights.json"
})

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize services
llm = None
vector_store = None
feedback_data = defaultdict(list)
model_weights = {
    "response_quality": 1.0,
    "relevance": 1.0,
    "helpfulness": 1.0
}

def initialize_llm():
    """Initialize the Groq LLM"""
    global llm
    try:
        llm = ChatGroq(
            temperature=0.1,
            model_name="llama3-70b-8192",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            request_timeout=120
        )
        return True
    except Exception as e:
        app.logger.error(f"LLM initialization failed: {str(e)}")
        return False

def load_model_weights():
    """Load model weights from file"""
    global model_weights
    try:
        if os.path.exists(app.config["MODEL_WEIGHTS_FILE"]):
            with open(app.config["MODEL_WEIGHTS_FILE"], "r") as f:
                model_weights = json.load(f)
    except Exception as e:
        app.logger.error(f"Failed to load model weights: {str(e)}")

def save_model_weights():
    """Save model weights to file"""
    try:
        with open(app.config["MODEL_WEIGHTS_FILE"], "w") as f:
            json.dump(model_weights, f, indent=2)
    except Exception as e:
        app.logger.error(f"Failed to save model weights: {str(e)}")

def update_model_weights(feedback):
    """Update model weights based on feedback"""
    # Simple reinforcement learning update
    learning_rate = 0.1
    
    if feedback["feedback"] == "positive":
        model_weights["response_quality"] = min(1.5, model_weights["response_quality"] + learning_rate)
        model_weights["relevance"] = min(1.5, model_weights["relevance"] + learning_rate)
        model_weights["helpfulness"] = min(1.5, model_weights["helpfulness"] + learning_rate)
    else:
        model_weights["response_quality"] = max(0.5, model_weights["response_quality"] - learning_rate)
        model_weights["relevance"] = max(0.5, model_weights["relevance"] - learning_rate)
        model_weights["helpfulness"] = max(0.5, model_weights["helpfulness"] - learning_rate)
    
    save_model_weights()

def apply_model_weights(prompt_template):
    """Apply model weights to the prompt"""
    weighted_template = f"""
    [AGENTIC MODE - QUALITY: {model_weights['response_quality']:.2f}] You are an AI assistant with autonomy. 
    Analyze the context and:
    1. Answer the question (RELEVANCE: {model_weights['relevance']:.2f})
    2. Summarize key points 
    3. Suggest related actions (HELPFULNESS: {model_weights['helpfulness']:.2f})
    
    Context: {{context}}
    
    Question: {{question}}
    
    Structured Response:
    """
    return weighted_template

# Load initial weights
load_model_weights()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "llm_initialized": llm is not None,
        "pdf_loaded": vector_store is not None,
        "model_weights": model_weights
    }
    return jsonify(status), 200

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and processing"""
    global vector_store
    
    if not initialize_llm():
        return jsonify({"error": "AI service initialization failed"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Sanitize filename
    safe_filename = "".join(c for c in file.filename if c.isalnum() or c in (' ', '.', '_')).rstrip()
    temp_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_filename)
    
    try:
        # Save the uploaded file
        file.save(temp_path)
        
        try:
            # Process PDF
            loader = PyPDFLoader(temp_path)
            documents = loader.load()
            
            # Split text with overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Store in FAISS
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            return jsonify({"message": "PDF processed successfully"})
            
        except Exception as e:
            app.logger.error(f"PDF processing error: {traceback.format_exc()}")
            return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
            
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                app.logger.warning(f"Could not delete temp file: {str(e)}")
                
    except Exception as e:
        app.logger.error(f"Upload error: {traceback.format_exc()}")
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat queries with the processed PDF"""
    global vector_store, llm, model_weights

    if not vector_store:
        return jsonify({"error": "No PDF loaded"}), 400
    if not llm:
        return jsonify({"error": "AI service unavailable"}), 503

    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing query parameter"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"error": "Empty query"}), 400

    try:
        # Configure retriever
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4},
        )

        # Get weighted prompt template
        prompt_template = apply_model_weights("""
        [AGENTIC MODE] You are an AI assistant with autonomy. 
        Analyze the context and:
        1. Answer the question
        2. Summarize key points
        3. Suggest related actions
        
        Context: {context}
        
        Question: {question}
        
        Structured Response:
        """)

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"],
                )
            },
            return_source_documents=True,
        )

        # Execute query
        result = qa_chain({"query": query})

        return jsonify({
            "response": result["result"],
            "sources": [
                {
                    "page": doc.metadata["page"] + 1,
                    "excerpt": doc.page_content[:100] + "...",
                }
                for doc in result["source_documents"]
            ],
            "model_weights": model_weights
        })

    except Exception as e:
        app.logger.error(f"Chat error: {traceback.format_exc()}")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route('/feedback', methods=['POST'])
def handle_feedback():
    """Handle user feedback and update model"""
    data = request.get_json()
    if not data or "prompt" not in data or "response" not in data or "feedback" not in data:
        return jsonify({"error": "Missing required feedback data"}), 400
    
    feedback_record = {
        "timestamp": datetime.now().isoformat(),
        "prompt": data["prompt"],
        "response": data["response"],
        "feedback": data["feedback"],
        "comments": data.get("comments", ""),
        "model_weights_before": model_weights.copy()
    }
    
    try:
        # Update model weights
        update_model_weights(data)
        feedback_record["model_weights_after"] = model_weights.copy()
        
        # Save feedback for analysis
        feedback_data[data["feedback"]].append(feedback_record)
        with open(app.config["FEEDBACK_FILE"], "a") as f:
            f.write(json.dumps(feedback_record) + "\n")
        
        return jsonify({
            "message": "Feedback processed and model updated",
            "new_weights": model_weights
        }), 200
        
    except Exception as e:
        app.logger.error(f"Feedback processing error: {str(e)}")
        return jsonify({"error": f"Failed to process feedback: {str(e)}"}), 500

def cleanup():
    """Clean up temporary directory"""
    try:
        if os.path.exists(app.config["UPLOAD_FOLDER"]):
            shutil.rmtree(app.config["UPLOAD_FOLDER"])
    except Exception as e:
        app.logger.warning(f"Could not clean up upload directory: {str(e)}")

# Register cleanup function
atexit.register(cleanup)

if __name__ == '__main__':
    # Ensure the upload directory exists
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    app.run(
        host="0.0.0.0",
        port=5000,
        threaded=True,
        debug=False
    )