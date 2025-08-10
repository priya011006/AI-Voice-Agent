import os
import tempfile
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

# ---------- Load env & ensure folders ----------
load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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
        safe_name = f"{int(__import__('time').time())}_{filename}"
        upload_path = os.path.join("uploads", safe_name)
        with open(upload_path, "wb") as buffer:
            shutil = __import__("shutil")
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

@app.post("/tts/echo/")
async def tts_echo(file: UploadFile = File(...)):
    """Echo Bot flow: audio upload → transcribe → generate TTS → return audio URL"""
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

# ---------- NEW: DAY 9 AI CONVERSATION ENDPOINT ----------
@app.post("/llm/voice-query/")
async def ai_voice_conversation(file: UploadFile = File(...)):
    """
    Day 9: Complete AI voice conversation pipeline
    1) Transcribe voice question with AssemblyAI
    2) Process with Gemini AI
    3) Generate voice response with Murf
    4) Return transcript, AI response, and audio URL
    """
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
        
        # Step 2: Query Gemini AI with the transcribed question
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
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            "transcript": user_question,
            "response": ai_text,
            "audio_url": response.audio_file
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"AI conversation failed: {str(e)}"})

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