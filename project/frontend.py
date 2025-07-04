import streamlit as st # type: ignore
import requests
import os
import time
from dotenv import load_dotenv
from gtts import gTTS # type: ignore
import base64
from io import BytesIO
import json

# --- PAGE CONFIG (MUST BE FIRST) ---
st.set_page_config(
    page_title="Smart Farming with AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD ENVIRONMENT ---
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5000")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Feedback section */
    .feedback-container {
        margin-top: 1rem;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .feedback-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    .model-stats {
        background-color: #e6f7ff;
        padding: 0.5rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    /* Rest of your existing CSS... */
</style>
""", unsafe_allow_html=True)

# --- AUDIO FUNCTIONS ---
def text_to_speech(text, lang='en'):
    """Convert text to speech audio"""
    tts = gTTS(text=text, lang=lang, slow=False)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes

def autoplay_audio(audio_bytes):
    """Create HTML audio player with autoplay"""
    audio_base64 = base64.b64encode(audio_bytes.read()).decode('utf-8')
    audio_html = f"""
    <audio controls autoplay class="audio-player">
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# --- INITIALIZE SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "upload_error" not in st.session_state:
    st.session_state.upload_error = None
if "audio_enabled" not in st.session_state:
    st.session_state.audio_enabled = False
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = {}
if "model_stats" not in st.session_state:
    st.session_state.model_stats = {
        "response_quality": 1.0,
        "relevance": 1.0,
        "helpfulness": 1.0
    }

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <h1 style='color: white; margin-bottom: 1.5rem;'>
        <span style='display: flex; align-items: center;'>
            🔄 Agentic AI-powered chatbot!
        </span>
    </h1>
    """, unsafe_allow_html=True)
    
    # Audio toggle
    st.session_state.audio_enabled = st.checkbox(
        "Enable Voice Responses",
        value=st.session_state.audio_enabled,
        help="Enable text-to-speech for AI responses"
    )
    
    # File Upload Section
    uploaded_file = st.file_uploader(
        "Upload PDF File",
        type=["pdf"],
        accept_multiple_files=False,
        help="Max file size: 200MB"
    )
    
    if uploaded_file is not None:
        try:
            # Check file size (200MB = 200 * 1024 * 1024 bytes)
            if uploaded_file.size > 200 * 1024 * 1024:
                st.error("❌ File size exceeds 200MB limit")
                st.stop()
                
            with st.spinner("Processing document (this may take a while for large files)..."):
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=300  # Increased timeout for large files
                )
                
            if response.status_code == 200:
                st.session_state.pdf_processed = True
                st.session_state.messages = []
                st.session_state.upload_error = None
                st.success("✅ Document processed successfully!")
                st.markdown("<div style='font-size: 2rem; text-align: center;'>🌱</div>", unsafe_allow_html=True)
            else:
                error_msg = response.json().get("error", "Unknown error")
                st.session_state.upload_error = error_msg
                st.error(f"❌ {error_msg}")
                
        except requests.exceptions.RequestException as e:
            st.session_state.upload_error = str(e)
            st.error(f"❌ Connection error: {str(e)}")

    # Model Statistics
    st.markdown("""
    <div style='margin-top: 2rem;'>
        <h2 style='color: white; margin-bottom: 0.5rem;'>Model Statistics</h2>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        health_response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.session_state.model_stats = health_data.get("model_weights", st.session_state.model_stats)
            
            st.markdown(f"""
            <div style='background-color: rgba(255, 255, 255, 0.2); padding: 0.75rem; border-radius: 10px; margin: 1rem 0;'>
                <p style='color: white; margin: 0.5rem 0;'>
                    <strong>Quality:</strong> 
                    <span style='color: {'#4CAF50' if st.session_state.model_stats['response_quality'] >= 1.0 else '#FF9800'};'>
                        {st.session_state.model_stats['response_quality']:.2f}
                    </span>
                </p>
                <p style='color: white; margin: 0.5rem 0;'>
                    <strong>Relevance:</strong> 
                    <span style='color: {'#4CAF50' if st.session_state.model_stats['relevance'] >= 1.0 else '#FF9800'};'>
                        {st.session_state.model_stats['relevance']:.2f}
                    </span>
                </p>
                <p style='color: white; margin: 0.5rem 0;'>
                    <strong>Helpfulness:</strong> 
                    <span style='color: {'#4CAF50' if st.session_state.model_stats['helpfulness'] >= 1.0 else '#FF9800'};'>
                        {st.session_state.model_stats['helpfulness']:.2f}
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error("⚠️ Backend unavailable")
    except requests.exceptions.RequestException:
        st.error("🔌 Cannot connect to backend")

# --- MAIN CHAT AREA ---
st.markdown("""
<h1 style='text-align: center; margin-bottom: 0.5rem;'>
    <span style='display: flex; justify-content: center; align-items: center;'>
        <span style='margin-right: 10px;'>🌱</span>
        Smart Farming with Unnati
    </span>
</h1>
<p style='text-align: center; color: #666; margin-top: 0;'>
                   Our Vision is
Empower farmers with digital technologies which
bring efficiencies in their farm business
</p>
""", unsafe_allow_html=True)

# Display chat messages with feedback
for i, message in enumerate(st.session_state.messages):
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
        if message.get("sources"):
            with st.expander("📚 Source References"):
                for source in message["sources"]:
                    st.markdown(f"**📄 Page {source['page']}:** {source['excerpt']}")
        
        # Add feedback for assistant messages (only last message)
        if message["role"] == "assistant" and i == len(st.session_state.messages) - 1:
            if i not in st.session_state.feedback_given:
                with st.container():
                    st.markdown("<div class='feedback-container'>", unsafe_allow_html=True)
                    st.markdown("**Help improve the AI:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("👍 Good Response", key=f"feedback_yes_{i}"):
                            try:
                                response = requests.post(
                                    f"{BACKEND_URL}/feedback",
                                    json={
                                        "prompt": st.session_state.messages[-2]["content"],
                                        "response": message["content"],
                                        "feedback": "positive"
                                    }
                                )
                                if response.status_code == 200:
                                    st.session_state.feedback_given[i] = True
                                    st.success("✅ Feedback saved - model updated!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save feedback")
                            except requests.exceptions.RequestException:
                                st.error("Connection error while saving feedback")
                    
                    with col2:
                        if st.button("👎 Needs Improvement", key=f"feedback_no_{i}"):
                            comments = st.text_input(
                                "What was wrong with this response?",
                                key=f"comments_{i}",
                                label_visibility="collapsed"
                            )
                            if comments:
                                try:
                                    response = requests.post(
                                        f"{BACKEND_URL}/feedback",
                                        json={
                                            "prompt": st.session_state.messages[-2]["content"],
                                            "response": message["content"],
                                            "feedback": "negative",
                                            "comments": comments
                                        }
                                    )
                                    if response.status_code == 200:
                                        st.session_state.feedback_given[i] = True
                                        st.success("✅ Feedback saved - model updated!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to save feedback")
                                except requests.exceptions.RequestException:
                                    st.error("Connection error while saving feedback")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask about the Unnati..."):
    if not st.session_state.pdf_processed:
        st.warning("⚠️ Please upload and process a PDF first")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
    
    # Get AI response
    try:
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing document..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": prompt},
                    timeout=120
                )
                
                if response.status_code == 200:
                    data = response.json()
                    full_response = data["response"]
                    
                    # Display text response
                    message_placeholder.markdown(full_response)
                    
                    # Update model stats
                    if "model_weights" in data:
                        st.session_state.model_stats = data["model_weights"]
                    
                    # Convert to audio if enabled
                    if st.session_state.audio_enabled:
                        try:
                            audio_bytes = text_to_speech(full_response)
                            autoplay_audio(audio_bytes)
                        except Exception as e:
                            st.warning(f"⚠️ Audio generation failed: {str(e)}")
                    
                    # Add sources if available
                    if data.get("sources"):
                        with st.expander("📚 Source References"):
                            for source in data["sources"]:
                                st.markdown(f"**📄 Page {source['page']}:** {source['excerpt']}")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": data.get("sources", [])
                    })
                else:
                    error_msg = response.json().get("error", "Unknown error")
                    st.error(f"❌ {error_msg}")
    except requests.exceptions.RequestException as e:
        st.error(f"🔌 Connection error: {str(e)}")

# Initial state instructions
if not st.session_state.pdf_processed and not st.session_state.upload_error:
    st.info("""
    **Getting Started:**
    1. Empowering Farmers with Digital Finance
    2. High-Quality Agri-Inputs at Your Fingertips
    3. Market Access for Better Profits
    4. Smart Farming with AI and Satellite Technology
    
   "Curious to know how Unnati is transforming agriculture? Discover the future of farming with us!"
    """)