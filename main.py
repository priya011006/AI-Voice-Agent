import os
import tempfile
import shutil
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import assemblyai as aai
from murf import Murf
import google.generativeai as genai
import time
from typing import List, Dict

# ---------- Load env & ensure folders ----------
load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# sanity checks (helpful error if keys missing)
if not MURF_API_KEY:
    print("WARNING: MURF_API_KEY not set in .env")
if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY not set in .env")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set in .env")

os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# ---------- FastAPI app ----------
app = FastAPI()

# CORS for frontend (local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static + uploads and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# ---------- Third-party clients ----------
# AssemblyAI
aai.settings.api_key = ASSEMBLYAI_API_KEY

# Murf
murf_client = Murf(api_key=MURF_API_KEY)

# Gemini
genai.configure(api_key=GEMINI_API_KEY)

# ---------- Models ----------
class TTSRequest(BaseModel):
    text: str

class LLMRequest(BaseModel):
    prompt: str

# ---------- Helper function for text chunking ----------
def chunk_text(text: str, max_length: int = 2800):
    """Chunk text for Murf API 3000 character limit"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
    current_chunk = ""
    
    for sentence in sentences:
        potential_chunk = current_chunk + ". " + sentence if current_chunk else sentence
        
        if len(potential_chunk) <= max_length:
            current_chunk = potential_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk + ".")
                current_chunk = sentence
            else:
                # Single sentence too long, truncate
                chunks.append(sentence[:max_length])
                current_chunk = ""
    
    if current_chunk:
        chunks.append(current_chunk + ".")
    
    return chunks

# ---------- In-memory chat history (Day 10) ----------
# WARNING: This is an in-memory store intended for prototype / single-worker use only.
chat_history: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_EXCHANGES = 8  # keep last N user+assistant messages (i.e. 16 entries max)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-audio/")
async def generate_audio(data: TTSRequest):
    """Generate TTS from text using Murf SDK."""
    try:
        client = murf_client
        response = client.text_to_speech.generate(
            text=data.text,
            voice_id="en-US-natalie",
            style="Promo",
            format="MP3"
        )
        return {"audio_url": response.audio_file}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to generate audio: {str(e)}"})

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    """Save uploaded audio to uploads/ for debugging or playback."""
    try:
        filename = file.filename
        safe_name = f"{int(time.time())}_{filename}"
        upload_path = os.path.join("uploads", safe_name)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(upload_path)
        return {
            "filename": safe_name,
            "content_type": file.content_type,
            "size": file_size,
            "url": f"/uploads/{safe_name}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Upload failed: {str(e)}"})

@app.post("/transcribe/file/")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file using AssemblyAI Python SDK."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)
        if transcript.status == aai.TranscriptStatus.error:
            return JSONResponse(status_code=500, content={"error": f"Transcription error: {transcript.error}"})

        return {"transcript": transcript.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Transcription failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

@app.post("/tts/echo/")
async def tts_echo(file: UploadFile = File(...)):
    """Echo Bot flow: audio upload → transcribe → generate TTS → return audio URL"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 1) Transcribe with AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)
        if transcript.status == aai.TranscriptStatus.error:
            return JSONResponse(status_code=500, content={"error": f"Transcription error: {transcript.error}"})
        text = transcript.text

        # 2) Generate Murf TTS from the transcript
        client = murf_client
        response = client.text_to_speech.generate(
            text=text,
            voice_id="en-US-natalie",
            style="Promo",
            format="MP3"
        )
        murf_audio_url = response.audio_file
        return {"transcript": text, "audio_url": murf_audio_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS echo failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------- Day 9 one-shot LLM endpoint (keep) ----------
@app.post("/llm/voice-query/")
async def ai_voice_conversation(file: UploadFile = File(...)):
    """
    Non-chat version preserved for compatibility.
    """
    tmp_path = None
    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Step 1: Transcribe the voice question
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)
        if transcript.status == aai.TranscriptStatus.error:
            return JSONResponse(status_code=500, content={"error": f"Transcription error: {transcript.error}"})
        
        user_question = transcript.text
        
        # Step 2: Query Gemini AI with the transcribed question (simple one-shot)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        ai_response = model.generate_content(user_question)
        ai_text = ai_response.text
        
        # Step 3: Chunk the AI response for Murf's character limit
        text_chunks = chunk_text(ai_text)
        
        # Use the first chunk for audio generation (or combine multiple if needed)
        text_for_audio = text_chunks[0] if text_chunks else ai_text[:2800]
        
        # Step 4: Generate Murf audio from AI response
        client = murf_client
        response = client.text_to_speech.generate(
            text=text_for_audio,
            voice_id="en-US-natalie",
            style="Promo",
            format="MP3"
        )
        
        return {
            "transcript": user_question,
            "response": ai_text,
            "audio_url": response.audio_file
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI conversation failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------- NEW: Day 10 - Conversational endpoint with chat history ----------
@app.post("/agent/chat/{session_id}")
async def agent_chat(session_id: str, file: UploadFile = File(...)):
    """
    Conversational flow with chat history stored in-memory keyed by session_id.
    Flow: audio -> STT -> append user message -> build prompt from history -> LLM -> append assistant message -> TTS -> return
    """
    tmp_path = None
    try:
        # 1) Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2) Transcribe
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(tmp_path)
        if transcript.status == aai.TranscriptStatus.error:
            return JSONResponse(status_code=500, content={"error": f"Transcription error: {transcript.error}"})
        user_text = transcript.text

        # 3) Append to chat history (create session if missing)
        history = chat_history.get(session_id, [])
        history.append({"role": "user", "content": user_text})

        # keep only last MAX_HISTORY_EXCHANGES * 2 messages
        # (we store messages as individual roles so count individual messages)
        if len(history) > MAX_HISTORY_EXCHANGES * 2:
            history = history[-MAX_HISTORY_EXCHANGES*2:]

        # 4) Build prompt for LLM by concatenating last messages
        # We'll build a conversational prompt that gives the model context.
        # Format: "User: ...\nAssistant: ...\nUser: ..."
        prompt_parts = []
        # Add a short system instruction so Gemini knows to behave as an assistant
        prompt_parts.append("You are a helpful assistant. Use previous messages for context when answering follow-up questions.")
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")

        # Add placeholder for assistant response
        prompt = "\n".join(prompt_parts) + "\nAssistant:"

        # 5) Query Gemini with full prompt (history included)
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        ai_response = model.generate_content(prompt)
        ai_text = ai_response.text

        # 6) Append assistant response to history
        history.append({"role": "assistant", "content": ai_text})
        # store back (trimmed)
        if len(history) > MAX_HISTORY_EXCHANGES * 2:
            history = history[-MAX_HISTORY_EXCHANGES*2:]
        chat_history[session_id] = history

        # 7) Generate TTS for assistant answer (first chunk)
        text_chunks = chunk_text(ai_text)
        text_for_audio = text_chunks[0] if text_chunks else ai_text[:2800]

        client = murf_client
        response = client.text_to_speech.generate(
            text=text_for_audio,
            voice_id="en-US-natalie",
            style="Promo",
            format="MP3"
        )

        return {
            "transcript": user_text,
            "response": ai_text,
            "audio_url": response.audio_file,
            "history": history
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Agent chat failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------- EXISTING GEMINI LLM ENDPOINT (Day 8) ----------
@app.post("/llm/query")
async def query_llm(request: LLMRequest):
    """Query Gemini LLM and return its response."""
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro")
        response = model.generate_content(request.prompt)
        return {"response": response.text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Gemini LLM failed: {str(e)}"})
