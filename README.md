# 🎙️ Pixie AI Voice Agent

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-blueviolet?logo=render)](https://render.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An **AI-powered Voice Agent** that converts text to speech, transcribes audio, and enables real-time interaction using modern AI APIs. Built with **FastAPI**, deployed on **Render**, and integrated with **Murf**, **AssemblyAI**, and **Gemini APIs**.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Gunicorn  
- **Frontend**: HTML, CSS, JavaScript  
- **APIs**: Murf (TTS), AssemblyAI (STT), Gemini (LLM), Tavily (Search)  
- **Deployment**: Render (Free Plan)  
- **Version Control**: Git & GitHub  

---

## ✨ Features

- 🔊 **Text-to-Speech** using Murf API  
- 🎧 **Speech-to-Text** transcription via AssemblyAI  
- 🤖 **Conversational AI** powered by Google Gemini  
- ⚡ **Real-time WebSocket** support for live interaction  
- 🌐 **Full-stack integration** with FastAPI backend and JS frontend  

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/priya011006/AI-Voice-Agent.git
cd AI-Voice-Agent

---

### 2. Create the virtual environment
python -m venv venv
source venv/bin/activate   # for Linux/Mac
venv\Scripts\activate      # for Windows


### **Install dependencies**
```bash
pip install -r requirements.txt
```

### **Set up environment variables**
Create a `.env` file in the project root with the following:
```env
GEMINI_API_KEY=your_gemini_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
MURF_API_KEY=your_murf_api_key
TAVILY_API_KEY=your_tavily_api_key

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

##📂 Project Structure

.
├── main.py                # FastAPI app
├── requirements.txt       # Dependencies
├── Procfile               # Deployment entrypoint
├── render.yaml            # Render deployment config
├── static/                # Frontend JS/CSS
│   └── script.js
├── templates/             # HTML templates
│   ├── index.html
│   └── websocket.html
├── uploads/               # Audio uploads
└── .env                   # Environment variables (ignored in git)

## 🚀 Deployment to Render.com

### Quick Deployment Steps

1. Sign up for a [Render account](https://render.com/)
2. Create a new Web Service and connect your GitHub repository
3. Configure the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Set the required environment variables in your Render dashboard:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `ASSEMBLYAI_API_KEY`: Your AssemblyAI API key
   - `MURF_API_KEY`: Your Murf API key
   - `TAVILY_API_KEY`: Your Tavily API key

---
)

##🤝 Contributing

Pull requests and suggestions are welcome!



##📜 License

This project is licensed under the MIT License.

---

⚡Pro tip: After pasting this into `README.md`, preview it in VS Code (right-click → **Open Preview**) to see how it will look on GitHub.  

Do you want me to also add **GitHub-style badges** (Python, FastAPI, Render, License, etc.) at the top for extra polish?


# AI-Voice-Agent
