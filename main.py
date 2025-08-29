import os
import tempfile
import shutil
from fastapi import FastAPI, Request, UploadFile, File, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import assemblyai as aai
import google.generativeai as genai
from google.generativeai import types
import time
from typing import List, Dict, Optional, Any
import logging
import json
import requests
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Load env & ensure folders ----------
load_dotenv()
MURF_API_KEY = os.getenv("MURF_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Dictionary to store user-provided API keys
user_api_keys = {
    "murf": MURF_API_KEY,
    "assemblyai": ASSEMBLYAI_API_KEY,
    "gemini": GEMINI_API_KEY,
    "tavily": TAVILY_API_KEY
}

# Check if we're in production (Render sets this environment variable)
IS_PRODUCTION = os.getenv("RENDER", "0") == "1"

# Log environment status
if IS_PRODUCTION:
    logger.info("Running in production mode")
else:
    logger.info("Running in development mode")
    if not MURF_API_KEY:
        logger.warning("MURF_API_KEY not set in .env")
    if not ASSEMBLYAI_API_KEY:
        logger.warning("ASSEMBLYAI_API_KEY not set in .env")
    if not GEMINI_API_KEY:
        logger.warning("GEMINI_API_KEY not set in .env")
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set in .env")

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

# ---------- Streaming LLM and TTS functions ----------
async def search_web(query: str) -> str:
    """Search the web using Tavily API."""
    try:
        api_key = user_api_keys.get("tavily", TAVILY_API_KEY)
        if not api_key:
            return "Web search is not available (Tavily API key not provided)."
            
        url = "https://api.tavily.com/search"
        params = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_domains": [],
            "exclude_domains": [],
            "max_results": 5
        }
        
        logger.info(f"Searching web for: {query}")
        response = requests.post(url, json=params)
        
        if response.status_code == 200:
            results = response.json()
            formatted_results = ""
            
            for i, result in enumerate(results.get("results", []), 1):
                title = result.get("title", "No title")
                content = result.get("content", "No content")
                url = result.get("url", "No URL")
                formatted_results += f"{i}. {title}\n{content}\nSource: {url}\n\n"
                
            return formatted_results if formatted_results else "No search results found."
        else:
            logger.error(f"Tavily API error: {response.status_code} - {response.text}")
            return f"Web search failed with status code: {response.status_code}"
            
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return f"Error searching web: {str(e)}"

async def get_weather(location: str) -> str:
    """Get weather information for a location."""
    try:
        # For demo purposes, return mock data based on location
        # In production, you would use a real weather API
        weather_data = {
            "new york": {"temp": "68°F", "condition": "Cloudy", "humidity": "70%", "wind": "8 mph"},
            "london": {"temp": "59°F", "condition": "Rainy", "humidity": "85%", "wind": "12 mph"},
            "tokyo": {"temp": "75°F", "condition": "Sunny", "humidity": "60%", "wind": "5 mph"},
            "paris": {"temp": "62°F", "condition": "Partly Cloudy", "humidity": "65%", "wind": "7 mph"},
            "sydney": {"temp": "82°F", "condition": "Clear", "humidity": "55%", "wind": "10 mph"},
        }
        
        # Default weather if location not in our mock data
        default_weather = {"temp": "72°F", "condition": "Partly Cloudy", "humidity": "65%", "wind": "6 mph"}
        
        # Get weather for the location (case insensitive)
        location_lower = location.lower()
        weather = weather_data.get(location_lower, default_weather)
        
        return f"Weather in {location}: {weather['temp']}, {weather['condition']}, Humidity: {weather['humidity']}, Wind: {weather['wind']}"
        
    except Exception as e:
        logger.error(f"Error getting weather: {str(e)}")
        return f"Error getting weather information: {str(e)}"

async def process_with_llm_and_tts(websocket: WebSocket, transcript: str):
    """Process transcript with streaming LLM and TTS."""
    try:
        logger.info(f"Processing transcript with LLM: {transcript}")
        
        # Configure Gemini for streaming
        api_key = user_api_keys.get("gemini", GEMINI_API_KEY)
        genai.configure(api_key=api_key)
        
        # Check if this is a web search query
        if "search" in transcript.lower() or "find information" in transcript.lower() or "look up" in transcript.lower():
            search_query = transcript.replace("search for", "").replace("search", "").replace("find information about", "").replace("look up", "").strip()
            
            # Inform user that search is happening
            await websocket.send_text(json.dumps({
                "type": "llm_chunk",
                "text": f"Searching the web for: {search_query}...",
                "accumulated": f"Searching the web for: {search_query}..."
            }))
            
            # Perform web search
            search_results = await search_web(search_query)
            
            # Define function for weather
            weather_function = {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                }
            }
            
            # Configure model with tools
            model = genai.GenerativeModel(
                'gemini-pro',
                tools=[types.Tool(function_declarations=[weather_function])]
            )
            
            # Create prompt with search results
            prompt = f"""You are Pixie, a helpful AI assistant with access to web search.
            
            User query: {transcript}
            
            Web search results:
            {search_results}
            
            Based on these search results, provide a helpful response to the user's query.
            If the user is asking about weather, use the get_weather function.
            """
            
        # Check if this is a weather query
        elif "weather" in transcript.lower() or "temperature" in transcript.lower() or "forecast" in transcript.lower() or "rain" in transcript.lower() or "sunny" in transcript.lower():
            # Define function for weather
            weather_function = {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city name, e.g. San Francisco",
                        },
                    },
                    "required": ["location"],
                }
            }
            
            # Configure model with tools
            model = genai.GenerativeModel(
                'gemini-pro',
                tools=[types.Tool(function_declarations=[weather_function])]
            )
            
            # Inform user that weather info is being retrieved
            await websocket.send_text(json.dumps({
                "type": "llm_chunk",
                "text": "Getting weather information...",
                "accumulated": "Getting weather information..."
            }))
            
            prompt = f"""You are Pixie, a helpful AI assistant with weather information capabilities.
            
            User query: {transcript}
            
            If the user is asking about weather, extract the location and use the get_weather function.
            Otherwise, respond normally to their query.
            """
        else:
            # Regular query - use standard model
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"""You are Pixie, a cute and helpful AI assistant.
            
            User query: {transcript}
            
            Respond in a friendly, helpful manner. Your personality is cheerful and supportive.
            """
        
        # Start streaming response from LLM
        accumulated_response = ""
        
        # Stream the response
        response = model.generate_content(
            prompt,
            stream=True,
        )
        
        # Check for function calls
        function_called = False
        
        # Send streaming updates to client
        for chunk in response:
            if hasattr(chunk, 'parts') and chunk.parts and hasattr(chunk.parts[0], 'function_call'):
                function_call = chunk.parts[0].function_call
                if function_call.name == "get_weather":
                    location = function_call.args["location"]
                    weather_info = await get_weather(location)
                    
                    # Send weather info to client
                    await websocket.send_text(json.dumps({
                        "type": "llm_chunk",
                        "text": f"\nWeather information: {weather_info}\n",
                        "accumulated": f"Weather information for {location}: {weather_info}"
                    }))
                    
                    accumulated_response = f"Weather information for {location}: {weather_info}"
                    function_called = True
            elif chunk.text:
                accumulated_response += chunk.text
                logger.info(f"LLM chunk: {chunk.text}")
                
                # Send LLM chunk to client
                await websocket.send_text(json.dumps({
                    "type": "llm_chunk",
                    "text": chunk.text,
                    "accumulated": accumulated_response
                }))
        
        logger.info(f"Final LLM response: {accumulated_response}")
        
        # Process with Murf TTS via WebSockets
        await generate_tts_with_murf(websocket, accumulated_response)
        
    except Exception as e:
        logger.error(f"Error processing with LLM: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": f"LLM processing error: {str(e)}"
        }))

# Static context ID for Murf WebSockets to avoid context limit errors
MURF_CONTEXT_ID = "voice-agent-session-1"

async def generate_tts_with_murf(websocket: WebSocket, text: str):
    """Generate TTS using Murf WebSockets."""
    import asyncio
    import websockets
    import base64
    
    try:
        logger.info(f"Generating TTS with Murf: {text}")
        
        # Murf WebSocket URL
        ws_url = f"wss://api.murf.ai/v1/speech/stream-input?api-key={MURF_API_KEY}&sample_rate=44100&channel_type=MONO&format=WAV&context_id={MURF_CONTEXT_ID}"
        
        async with websockets.connect(ws_url) as murf_ws:
            # Send voice configuration
            voice_config = {
                "voice_config": {
                    "voiceId": "en-US-amara",
                    "style": "Conversational",
                    "rate": 0,
                    "pitch": 0,
                    "variation": 1
                }
            }
            
            logger.info(f"Sending voice config to Murf")
            await murf_ws.send(json.dumps(voice_config))
            
            # Send text for TTS
            text_msg = {
                "text": text,
                "end": True  # Close the context
            }
            
            logger.info(f"Sending text to Murf")
            await murf_ws.send(json.dumps(text_msg))
            
            # Receive audio chunks
            while True:
                response = await murf_ws.recv()
                data = json.loads(response)
                
                logger.info(f"Received Murf response: {data.keys()}")
                
                if "audio" in data:
                    # Log base64 audio (first 100 chars)
                    audio_preview = data["audio"][:100] + "..." if len(data["audio"]) > 100 else data["audio"]
                    logger.info(f"Received audio from Murf: {audio_preview}")
                    
                    # Send audio to client
                    await websocket.send_text(json.dumps({
                        "type": "tts_audio",
                        "audio": data["audio"]  # Base64 encoded audio
                    }))
                
                # Check if this is the final chunk
                if data.get("final", False):
                    logger.info("Received final audio chunk from Murf")
                    break
                    
    except Exception as e:
        logger.error(f"Error generating TTS with Murf: {str(e)}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": f"TTS generation error: {str(e)}"
        }))

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

# ---------- Fallback TTS Function ----------
def generate_fallback_audio(error_message: str = "I'm having trouble connecting right now. Please try again later."):
    """Generate a fallback audio response when TTS fails"""
    try:
        # Try with backup TTS settings or simplified request
        if MURF_API_KEY and murf_client:
            response = murf_client.text_to_speech.generate(
                text=error_message,
                voice_id="en-US-natalie",
                style="Conversational",  # Use simpler style
                format="MP3"
            )
            return response.audio_file
    except Exception as e:
        logger.error(f"Fallback TTS also failed: {e}")
    
    # Return a data URL for a simple beep or error sound if TTS completely fails
    # This is a simple base64 encoded silent audio file as absolute fallback
    return "data:audio/mp3;base64,SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU4Ljc2LjEwMAAAAAAAAAAAAAAA//tQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA4ODg4ODg4ODg4ODg4ODg4ODg4ODg4ODg4OD//////////////////////////////////////////////////////////////////8AAAAATGF2YzU4LjEzAAAAAAAAAAAAAAAAJAAAAAAAAAAAAAAAAAAAAAAAAAAA"

# ---------- Enhanced TTS Function with Error Handling ----------
def safe_generate_tts(text: str, max_retries: int = 2):
    """Generate TTS with fallback handling"""
    if not text.strip():
        return generate_fallback_audio("I didn't receive any text to speak.")
    
    for attempt in range(max_retries + 1):
        try:
            if not MURF_API_KEY:
                raise Exception("Murf API key not configured")
            
            client = murf_client
            response = client.text_to_speech.generate(
                text=text,
                voice_id="en-US-natalie",
                style="Promo",
                format="MP3"
            )
            logger.info(f"TTS successful on attempt {attempt + 1}")
            return response.audio_file
            
        except Exception as e:
            logger.warning(f"TTS attempt {attempt + 1} failed: {e}")
            if attempt == max_retries:
                # Last attempt failed, use fallback
                logger.error("All TTS attempts failed, using fallback")
                return generate_fallback_audio("I'm having trouble with my voice right now, but I'm still here to help you.")
    
    return generate_fallback_audio()

# ---------- In-memory chat history (Day 10) ----------
# WARNING: This is an in-memory store intended for prototype / single-worker use only.
chat_history: Dict[str, List[Dict[str, str]]] = {}
MAX_HISTORY_EXCHANGES = 8  # keep last N user+assistant messages (i.e. 16 entries max)

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/websocket-test", response_class=HTMLResponse)
async def websocket_test(request: Request):
    """WebSocket test page for demonstrating real-time communication."""
    return templates.TemplateResponse("websocket.html", {"request": request})

@app.post("/generate-audio/")
async def generate_audio(data: TTSRequest):
    """Generate TTS from text using Murf SDK with error handling."""
    try:
        logger.info(f"Generating TTS for text: {data.text[:50]}...")
        
        if not data.text or not data.text.strip():
            return JSONResponse(status_code=400, content={"error": "Text is required for audio generation"})
        
        # Use the safe TTS function with fallback
        audio_url = safe_generate_tts(data.text)
        return {"audio_url": audio_url}
        
    except Exception as e:
        logger.error(f"Unexpected error in generate_audio: {e}")
        # Even if everything fails, provide a fallback
        fallback_url = generate_fallback_audio("Sorry, I'm having technical difficulties generating audio right now.")
        return {"audio_url": fallback_url, "warning": "Using fallback audio due to technical issues"}

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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            logger.info(f"Received message: {data}")
            
            try:
                message = json.loads(data)
                
                if message.get("type") == "audio":
                    # Process audio data
                    audio_data = message["audio"]
                    
                    # Remove the data:audio/webm;base64, prefix if present
                    if "base64," in audio_data:
                        audio_data = audio_data.split("base64,")[1]
                    
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_data)
                    
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(audio_bytes)
                    
                    # Transcribe with AssemblyAI
                    try:
                        # Use user-provided API key if available
                        api_key = user_api_keys.get("assemblyai", ASSEMBLYAI_API_KEY)
                        aai.settings.api_key = api_key
                        
                        transcriber = aai.Transcriber()
                        transcript = transcriber.transcribe(temp_file_path)
                        
                        if transcript.status == "completed":
                            # Send transcript to client
                            await websocket.send_text(json.dumps({
                                "type": "transcript",
                                "text": transcript.text
                            }))
                            
                            # Process with LLM and TTS
                            await process_with_llm_and_tts(websocket, transcript.text)
                        else:
                            await websocket.send_text(json.dumps({
                                "type": "error",
                                "error": f"Transcription failed: {transcript.status}"
                            }))
                    except Exception as e:
                        logger.error(f"Error transcribing audio: {str(e)}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": f"Transcription error: {str(e)}"
                        }))
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_file_path):
                            os.unlink(temp_file_path)
                
                elif message.get("type") == "config":
                    # Update API keys from client
                    logger.info("Received API key configuration")
                    
                    if "api_keys" in message:
                        api_keys = message["api_keys"]
                        
                        # Update API keys
                        for key_name, key_value in api_keys.items():
                            if key_value and key_name in user_api_keys:
                                user_api_keys[key_name] = key_value
                                logger.info(f"Updated {key_name} API key")
                        
                        # Update client libraries with new keys
                        if "assemblyai" in api_keys and api_keys["assemblyai"]:
                            aai.settings.api_key = api_keys["assemblyai"]
                        
                        if "murf" in api_keys and api_keys["murf"]:
                            global murf_client
                            murf_client = Murf(api_key=api_keys["murf"])
                        
                        if "gemini" in api_keys and api_keys["gemini"]:
                            genai.configure(api_key=api_keys["gemini"])
                        
                        # Acknowledge receipt of API keys
                        await websocket.send_text(json.dumps({
                            "type": "config_response",
                            "status": "success",
                            "message": "API keys updated successfully"
                        }))
                
                elif message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                else:
                    # Echo the message back to the client for other message types
                    response = f"Server received: {data}"
                    await websocket.send_text(response)
                    logger.info(f"Sent response: {response}")
            except json.JSONDecodeError:
                # Handle plain text messages
                response = f"Server received: {data}"
                await websocket.send_text(response)
                logger.info(f"Sent response: {response}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # Handle disconnection
        logger.info("WebSocket connection closed")

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming audio data with real-time transcription."""
    await websocket.accept()
    
    # Initialize AssemblyAI transcriber with turn detection
    transcriber = aai.RealtimeTranscriber(
        sample_rate=16000,  # AssemblyAI expects 16kHz audio
        word_boost=["hello", "hi", "hey"],  # Optional: boost recognition of these words
        encoding="pcm_s16le",  # 16-bit PCM encoding
        end_of_turn_confidence_threshold=0.7,  # Confidence threshold for turn detection
        format_turns=True,  # Enable formatted turns
    )
    
    # Track the current transcript for LLM processing
    current_transcript = ""
    final_transcript = ""
    
    # Set up the transcription handler
    @transcriber.on_data
    async def on_data(transcript):
        """Handle incoming transcription data."""
        nonlocal current_transcript, final_transcript
        
        if transcript.text:
            logger.info(f"Transcription: {transcript.text} (end_of_turn: {transcript.end_of_turn})")
            current_transcript = transcript.text
            
            # Send transcription back to client
            await websocket.send_text(json.dumps({
                "type": "transcription", 
                "text": transcript.text,
                "is_final": transcript.is_final,
                "end_of_turn": transcript.end_of_turn
            }))
            
            # If end of turn is detected, process with LLM
            if transcript.end_of_turn:
                logger.info(f"End of turn detected: {transcript.text}")
                final_transcript = transcript.text
                
                # Notify client about end of turn
                await websocket.send_text(json.dumps({
                    "type": "turn_end",
                    "final_transcript": final_transcript
                }))
                
                # Process with LLM and stream response
                await process_with_llm_and_tts(websocket, final_transcript)
    
    @transcriber.on_error
    async def on_error(error):
        """Handle transcription errors."""
        logger.error(f"Transcription error: {error}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "error": str(error)
        }))
    
    try:
        # Create a unique filename for this session
        filename = f"uploads/stream_{int(time.time())}.webm"
        logger.info(f"Starting audio stream to {filename}")
        
        # Start the transcriber
        await transcriber.start()
        logger.info("AssemblyAI transcriber started")
        
        with open(filename, "wb") as audio_file:
            while True:
                # Receive binary data from client
                data = await websocket.receive_bytes()
                logger.info(f"Received audio chunk: {len(data)} bytes")
                
                # Write the audio chunk to file
                audio_file.write(data)
                
                # Send audio data to AssemblyAI for transcription
                await transcriber.send_audio(data)
                
                # Send acknowledgment back to client
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "status": "received", 
                    "bytes": len(data)
                }))
    except Exception as e:
        logger.error(f"WebSocket audio error: {e}")
    finally:
        # Close the transcriber
        await transcriber.close()
        logger.info("AssemblyAI transcriber closed")
        
        # Handle disconnection
        logger.info("WebSocket audio connection closed")

@app.post("/transcribe/file/")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file using AssemblyAI Python SDK with error handling."""
    tmp_path = None
    try:
        logger.info(f"Starting transcription for file: {file.filename}")
        
        # Validate file
        if not file or not file.filename:
            return JSONResponse(status_code=400, content={"error": "No audio file provided"})
        
        # Check API key
        if not ASSEMBLYAI_API_KEY:
            return JSONResponse(status_code=500, content={
                "error": "Speech-to-text service unavailable",
                "fallback_message": "I'm having trouble understanding audio right now. Please try typing your message instead."
            })
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Attempt transcription with retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(tmp_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    error_msg = f"Transcription error: {transcript.error}"
                    if attempt == max_retries:
                        return JSONResponse(status_code=500, content={
                            "error": error_msg,
                            "fallback_message": "I couldn't understand the audio. Could you try speaking more clearly or check your microphone?"
                        })
                    continue
                
                logger.info(f"Transcription successful on attempt {attempt + 1}")
                return {"transcript": transcript.text}
                
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    return JSONResponse(status_code=500, content={
                        "error": f"Transcription failed after {max_retries + 1} attempts: {str(e)}",
                        "fallback_message": "I'm having trouble processing audio right now. Please try again or use text input instead."
                    })
                
    except Exception as e:
        logger.error(f"Unexpected error in transcribe_file: {e}")
        return JSONResponse(status_code=500, content={
            "error": f"Transcription service error: {str(e)}",
            "fallback_message": "There's a technical issue with audio processing. Please try again later."
        })
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
    user_text = ""
    ai_text = ""
    
    try:
        logger.info(f"Starting chat for session: {session_id}")
        
        # 1) Save uploaded audio temporarily
        if not file or not file.filename:
            return JSONResponse(status_code=400, content={
                "error": "No audio file provided",
                "fallback_audio": generate_fallback_audio("I didn't receive any audio. Please try recording again.")
            })
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".webm") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # 2) Transcribe with error handling
        try:
            if not ASSEMBLYAI_API_KEY:
                raise Exception("AssemblyAI API key not configured")
                
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(tmp_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                error_msg = f"Transcription error: {transcript.error}"
                logger.error(error_msg)
                return JSONResponse(status_code=500, content={
                    "error": error_msg,
                    "fallback_audio": generate_fallback_audio("I'm having trouble understanding your audio. Could you try speaking more clearly?")
                })
            user_text = transcript.text
            logger.info(f"Transcription successful: {user_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return JSONResponse(status_code=500, content={
                "error": f"Speech-to-text failed: {str(e)}",
                "fallback_audio": generate_fallback_audio("I'm having trouble with my hearing right now. Please try again or use text input.")
            })

        # 3) Append to chat history (create session if missing)
        history = chat_history.get(session_id, [])
        history.append({"role": "user", "content": user_text})

        # keep only last MAX_HISTORY_EXCHANGES * 2 messages
        if len(history) > MAX_HISTORY_EXCHANGES * 2:
            history = history[-MAX_HISTORY_EXCHANGES*2:]

        # 4) Build prompt for LLM
        try:
            prompt_parts = []
            prompt_parts.append("You are a helpful AI assistant in a voice conversation. Remember information from previous messages in this conversation and refer to it when relevant. Be conversational and natural.")
            prompt_parts.append("")
            
            # Build conversation history - exclude current message
            for i, msg in enumerate(history[:-1]):
                role = msg.get("role")
                content = msg.get("content")
                if role == "user":
                    prompt_parts.append(f"Human: {content}")
                else:
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt_parts.append(f"Human: {user_text}")
            prompt_parts.append("Assistant:")
            prompt = "\n".join(prompt_parts)
            
            logger.info(f"Session {session_id}: History length: {len(history)}")

        except Exception as e:
            logger.error(f"Failed to build prompt: {e}")
            # Continue with simple prompt
            prompt = f"You are a helpful AI assistant. Human: {user_text}\nAssistant:"

        # 5) Query Gemini with error handling
        try:
            if not GEMINI_API_KEY:
                raise Exception("Gemini API key not configured")
                
            model = genai.GenerativeModel("models/gemini-2.5-pro")
            ai_response = model.generate_content(prompt)
            ai_text = ai_response.text
            logger.info("LLM response generated successfully")
            
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            ai_text = "I'm having trouble thinking right now due to a technical issue. Could you please try asking your question again?"
            
            # Return with fallback audio
            fallback_audio = generate_fallback_audio(ai_text)
            return JSONResponse(status_code=500, content={
                "error": f"AI service unavailable: {str(e)}",
                "transcript": user_text,
                "response": ai_text,
                "audio_url": fallback_audio,
                "fallback_used": True
            })

        # 6) Append assistant response to history
        history.append({"role": "assistant", "content": ai_text})
        if len(history) > MAX_HISTORY_EXCHANGES * 2:
            history = history[-MAX_HISTORY_EXCHANGES*2:]
        chat_history[session_id] = history

        # 7) Generate TTS with error handling
        try:
            text_chunks = chunk_text(ai_text)
            text_for_audio = text_chunks[0] if text_chunks else ai_text[:2800]
            
            audio_url = safe_generate_tts(text_for_audio)
            
            return {
                "transcript": user_text,
                "response": ai_text,
                "audio_url": audio_url,
                "history": history,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            # Use fallback audio but still return the text response
            fallback_audio = generate_fallback_audio("I have a response for you, but I'm having trouble speaking right now.")
            return {
                "transcript": user_text,
                "response": ai_text,
                "audio_url": fallback_audio,
                "history": history,
                "session_id": session_id,
                "warning": "Audio generation failed, using fallback"
            }

    except Exception as e:
        logger.error(f"Unexpected error in agent_chat: {e}")
        fallback_audio = generate_fallback_audio("I'm experiencing technical difficulties. Please try again in a moment.")
        
        return JSONResponse(status_code=500, content={
            "error": f"Agent chat failed: {str(e)}",
            "transcript": user_text or "Could not transcribe audio",
            "response": "I'm sorry, I'm having technical problems right now.",
            "audio_url": fallback_audio,
            "fallback_used": True
        })
        
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------- DEBUG ENDPOINT ----------
@app.get("/debug/chat/{session_id}")
async def debug_chat_history(session_id: str):
    """Debug endpoint to view chat history for a session."""
    history = chat_history.get(session_id, [])
    return {
        "session_id": session_id,
        "history": history,
        "total_messages": len(history),
        "all_sessions": list(chat_history.keys())
    }

@app.get("/debug/api-status")
async def debug_api_status():
    """Debug endpoint to check API key status and simulate failures."""
    return {
        "murf_api_key": "configured" if MURF_API_KEY else "missing",
        "assemblyai_api_key": "configured" if ASSEMBLYAI_API_KEY else "missing", 
        "gemini_api_key": "configured" if GEMINI_API_KEY else "missing",
        "note": "To test error scenarios, comment out API keys in your .env file"
    }

@app.post("/debug/simulate-error/{error_type}")
async def simulate_error(error_type: str):
    """Simulate different types of errors for testing."""
    if error_type == "stt":
        return JSONResponse(status_code=500, content={
            "error": "Simulated STT failure",
            "fallback_audio": generate_fallback_audio("This is a simulated speech-to-text error for testing.")
        })
    elif error_type == "llm":
        return JSONResponse(status_code=500, content={
            "error": "Simulated LLM failure", 
            "response": "I'm simulating an AI thinking problem for testing purposes.",
            "audio_url": generate_fallback_audio("This is a simulated AI error for testing."),
            "fallback_used": True
        })
    elif error_type == "tts":
        return {
            "transcript": "Test message",
            "response": "This response should have normal text but fallback audio.",
            "audio_url": generate_fallback_audio("This is a simulated text-to-speech error for testing."),
            "warning": "TTS simulation failed"
        }
    else:
        return JSONResponse(status_code=400, content={"error": "Invalid error type. Use: stt, llm, or tts"})

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
