// ================= Text-to-Speech (Generate Audio) =================
document.getElementById("submit-button").addEventListener("click", async () => {
  const textInput = document.getElementById("text-input").value.trim();
  const audioPlayer = document.getElementById("audio-player");
  const statusText = document.getElementById("status-text");

  if (!textInput) {
    statusText.textContent = "⚠️ Please enter some text.";
    return;
  }

  statusText.textContent = "🎤 Generating audio...";

  try {
    const response = await fetch("/generate-audio/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textInput }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || "Failed to generate audio.");
    }

    const result = await response.json();
    audioPlayer.src = result.audio_url;
    audioPlayer.style.display = "block";
    statusText.textContent = "✅ Audio generated successfully!";
  } catch (error) {
    console.error("TTS Error:", error);
    statusText.textContent = "❌ Failed to generate audio. See console.";
  }
});

// ================= Echo Bot v2 (Recording → Transcribe → Murf → Play) =================
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

const recordBtn = document.getElementById("record-btn");
const stopBtn = document.getElementById("stop-btn");
const echoAudio = document.getElementById("echo-audio");
const transcriptionText = document.getElementById("transcription-text");

recordBtn.addEventListener("click", async () => {
  if (isRecording) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
      const file = new File([audioBlob], "recording.webm", { type: "audio/webm" });

      const formData = new FormData();
      formData.append("file", file);

      transcriptionText.innerText = "⏳ Uploading, transcribing, and generating Murf audio...";

      try {
        const resp = await fetch("/tts/echo/", {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || "Echo bot failed");
        }

        const result = await resp.json();

        transcriptionText.innerText = `📝 Transcription: ${result.transcript}`;
        if (result.audio_url) {
          echoAudio.src = result.audio_url;
          echoAudio.style.display = "block";
          echoAudio.oncanplay = () => echoAudio.play().catch(()=>{});
        } else {
          transcriptionText.innerText += " (No audio_url returned)";
        }
      } catch (error) {
        console.error("Echo Bot Error:", error);
        transcriptionText.innerText = "❌ Failed to process audio. See console.";
      }
    };

    mediaRecorder.start();
    isRecording = true;
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    transcriptionText.innerText = "🎙️ Recording...";
  } catch (err) {
    console.error("Recording Error:", err);
    transcriptionText.innerText = "⚠️ Mic permission denied or unavailable.";
  }
});

stopBtn.addEventListener("click", () => {
  if (!isRecording) return;
  mediaRecorder.stop();
  isRecording = false;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  transcriptionText.innerText = "⏳ Processing...";
});

// ================= NEW: AI Conversation (Day 9) =================
let aiMediaRecorder;
let aiAudioChunks = [];
let isAIRecording = false;

const aiRecordBtn = document.getElementById("ai-record-btn");
const aiStopBtn = document.getElementById("ai-stop-btn");
const aiAudio = document.getElementById("ai-audio");
const aiStatusText = document.getElementById("ai-status-text");
const conversationDisplay = document.getElementById("conversation-display");
const userQuestionDisplay = document.getElementById("user-question-display");
const aiResponseDisplay = document.getElementById("ai-response-display");

aiRecordBtn.addEventListener("click", async () => {
  if (isAIRecording) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    aiMediaRecorder = new MediaRecorder(stream);
    aiAudioChunks = [];

    aiMediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) aiAudioChunks.push(e.data);
    };

    aiMediaRecorder.onstop = async () => {
      const audioBlob = new Blob(aiAudioChunks, { type: "audio/webm" });
      const file = new File([audioBlob], "ai_question.webm", { type: "audio/webm" });

      const formData = new FormData();
      formData.append("file", file);

      aiStatusText.innerText = "🧠 Processing your question with AI...";

      try {
        const resp = await fetch("/llm/voice-query/", {
          method: "POST",
          body: formData,
        });

        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.error || "AI conversation failed");
        }

        const result = await resp.json();

        // Display conversation
        conversationDisplay.style.display = "block";
        userQuestionDisplay.innerHTML = `<strong>Your Question:</strong> ${result.transcript}`;
        aiResponseDisplay.innerHTML = `<strong>AI Response:</strong> ${result.response}`;
        
        aiStatusText.innerText = "✅ Conversation complete! Ask another question anytime.";
        
        // Play AI response audio
        if (result.audio_url) {
          aiAudio.src = result.audio_url;
          aiAudio.style.display = "block";
          aiAudio.oncanplay = () => aiAudio.play().catch(()=>{});
        }
      } catch (error) {
        console.error("AI Conversation Error:", error);
        aiStatusText.innerText = "❌ Failed to process AI conversation. See console.";
      }
    };

    aiMediaRecorder.start();
    isAIRecording = true;
    aiRecordBtn.disabled = true;
    aiStopBtn.disabled = false;
    aiStatusText.innerText = "🎙️ Recording your question...";
  } catch (err) {
    console.error("AI Recording Error:", err);
    aiStatusText.innerText = "⚠️ Mic permission required for AI conversation.";
  }
});

aiStopBtn.addEventListener("click", () => {
  if (!isAIRecording) return;
  aiMediaRecorder.stop();
  isAIRecording = false;
  aiRecordBtn.disabled = false;
  aiStopBtn.disabled = true;
  aiStatusText.innerText = "⏳ Processing your question...";
});