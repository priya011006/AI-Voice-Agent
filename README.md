
# 30 Days of AI Voice Agents

This repository documents my 30-day challenge of building AI-powered voice agents using **FastAPI**, **Murf TTS**, and **AssemblyAI**.  
Each day, I focus on implementing a new feature or improvement, sharing my progress along the way.

---

## 🛠 Tech Stack
- **Backend**: FastAPI (Python)
- **Frontend**: HTML, CSS, JavaScript
- **TTS (Text-to-Speech)**: Murf.ai API / Murf Python SDK
- **Speech-to-Text (Transcription)**: AssemblyAI API
- **Version Control**: Git & GitHub

---

## 📅 Progress Log

### **Day 1** — FastAPI Setup & Serving HTML
- Set up a basic FastAPI project.
- Served an HTML template with basic UI.
- Folder structure created:


---

### **Day 2** — TTS REST API Call with Murf
- Integrated Murf.ai API using the Murf Python SDK.
- Created `/generate-audio/` POST endpoint.
- Generated audio from user text input and returned an audio URL.

---

### **Day 3** — Real-time TTS Playback
- Sent text from the frontend to the backend.
- Played the generated audio instantly in the browser.
- Improved UI design with clean and modern styling.

---

### **Day 4** — Echo Bot Recording & Playback
- Added a voice recorder in the UI.
- Created `/upload-audio/` endpoint to receive recorded audio.
- Saved uploaded audio in `/uploads` folder.
- Played back the same audio as an echo.

---

### **Day 5** — Audio Upload Metadata
- Enhanced `/upload-audio/` to return:
- File name
- Content type
- File size (KB)
- Displayed metadata in the UI after upload.

---
### **Day 6** — Audio Transcription from Binary
- Created `/transcribe/file` endpoint.
- Accepted recorded audio directly as binary without saving to disk.
- Used **AssemblyAI API** to transcribe audio.
- Returned the transcription to the frontend.

---

### **Day 7** — Transcription Display in UI & TTS Echo Bot
- Improved UI to **display transcription neatly below the Echo Bot** section.
- Added transcription card with clear heading and style.
- Automatically shows:
  - "Transcribing..." while processing.
  - Final transcription result when ready.
- Implemented Echo Bot: uploaded audio is transcribed and then spoken back using Murf TTS.

---

### **Day 8** — Large Language Model (LLM) Integration
- Added a new `/llm/query` POST endpoint to the backend.
- Integrated with **Google Gemini API** for LLM-powered text generation.
- The endpoint accepts text prompts and returns LLM responses.
- No UI changes needed; backend is ready for advanced conversational AI.

---

Absolutely! Here's your Day 9 README update in the same clean and consistent format:

---

### **Day 9** — Full Non-Streaming Voice Agent Pipeline  
- Upgraded `/llm/query` endpoint to **accept audio input** directly.  
- On "Stop recording" click:  
- Audio is sent to backend for transcription.  
- Transcribed text is passed to **Google Gemini API** for LLM response.  
- LLM response is converted to speech using **Murf TTS**.  
- Final audio response is returned and played in the `<audio>` element on the UI.  
- Achieved a complete voice interaction loop — from user speech to LLM reply, all spoken back with Murf voice.  

---



 ## 🚀 Running the Project Locally

### **Clone the repository**
```bash
git clone https://github.com/yourusername/30-days-ai-voice-agents.git
cd 30-days-ai-voice-agents
```

### **Install dependencies**
```bash
pip install -r requirements.txt
```

### **Set up environment variables**
Create a `.env` file in the project root with the following:
```env
ASSEMBLYAI_API_KEY=your-assemblyai-api-key
MURF_API_KEY=your-murf-api-key
GEMINI_API_KEY=your-gemini-api-key
```
*(Do not share your API keys publicly.)*

### **Run the FastAPI server**
```bash
uvicorn main:app --reload
```

### **Open the app**
- Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser for the UI.
- Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for API documentation and testing.


## 🛠️ Endpoints Overview

- `POST /transcribe/file` — AssemblyAI-powered audio transcription.
- `POST /generate-audio` — Murf TTS: generate audio from text.
- `POST /tts/echo` — Echo bot: transcribe then speak back.
- `POST /llm/query` — Query Gemini LLM with a prompt.

---

## 💡 Tech Used

- FastAPI & Python
- AssemblyAI (Speech-to-Text)
- Murf (Text-to-Speech)
- Google Gemini (LLM)
- JavaScript, HTML, CSS (Frontend)

# AI-Voice-Agent
