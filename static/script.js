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
let lastStream = null;

const recordBtn = document.getElementById("record-btn");
const stopBtn = document.getElementById("stop-btn");
const echoAudio = document.getElementById("echo-audio");
const transcriptionText = document.getElementById("transcription-text");

async function startLocalRecording(onStopCallback, fileName = "recording.webm") {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  lastStream = stream;
  const mr = new MediaRecorder(stream);
  let chunks = [];

  mr.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  mr.onstop = async () => {
    const audioBlob = new Blob(chunks, { type: "audio/webm" });
    const file = new File([audioBlob], fileName, { type: "audio/webm" });
    // release tracks
    try { stream.getTracks().forEach(t => t.stop()); } catch(e){}
    onStopCallback(file);
  };

  mr.start();
  return mr;
}

recordBtn.addEventListener("click", async () => {
  if (isRecording) return;

  try {
    mediaRecorder = await startLocalRecording(async (file) => {
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
    });

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
  if (!isRecording || !mediaRecorder) return;
  mediaRecorder.stop();
  isRecording = false;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  transcriptionText.innerText = "⏳ Processing...";
});

// ================= NEW: AI Conversation (Day 9 -> Day 10 update) =================
let aiMediaRecorder;
let aiAudioChunks = [];
let isAIRecording = false;
let lastAIStream = null;

const aiRecordBtn = document.getElementById("ai-record-btn");
const aiStopBtn = document.getElementById("ai-stop-btn");
const aiAudio = document.getElementById("ai-audio");
const aiStatusText = document.getElementById("ai-status-text");
const conversationDisplay = document.getElementById("conversation-display");
const userQuestionDisplay = document.getElementById("user-question-display");
const aiResponseDisplay = document.getElementById("ai-response-display");

// Helper: get or create session id in URL query
function getSessionId() {
  const urlParams = new URLSearchParams(window.location.search);
  let session = urlParams.get('session');
  if (!session) {
    session = Math.random().toString(36).substring(2, 10);
    urlParams.set('session', session);
    const newUrl = `${location.pathname}?${urlParams.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }
  return session;
}

const SESSION_ID = getSessionId();

// Start recording helper (for AI conversation) — re-usable for auto-start
async function startAIRecording(fileName = "ai_question.webm") {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    lastAIStream = stream;
    aiMediaRecorder = new MediaRecorder(stream);
    aiAudioChunks = [];

    aiMediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) aiAudioChunks.push(e.data);
    };

    aiMediaRecorder.onstop = async () => {
      const audioBlob = new Blob(aiAudioChunks, { type: "audio/webm" });
      const file = new File([audioBlob], fileName, { type: "audio/webm" });

      const formData = new FormData();
      formData.append("file", file);

      aiStatusText.innerText = "🧠 Processing your question with AI...";

      try {
        const resp = await fetch(`/agent/chat/${SESSION_ID}`, {
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

        // Play AI response audio and after it ends, auto-start recording again
        if (result.audio_url) {
          aiAudio.src = result.audio_url;
          aiAudio.style.display = "block";
          aiAudio.oncanplay = () => aiAudio.play().catch(()=>{});

          // when finished playing — start a new recording automatically
          aiAudio.onended = () => {
            // small delay to ensure UI updates & avoid immediate re-trigger issues
            setTimeout(() => {
              // auto start recording again
              aiRecordBtn.disabled = true;
              aiStopBtn.disabled = false;
              aiStatusText.innerText = "🎙️ Listening for your next question...";
              startAIRecording();
            }, 700);
          };
        }

      } catch (error) {
        console.error("AI Conversation Error:", error);
        aiStatusText.innerText = "❌ Failed to process AI conversation. See console.";
      }

      // release microphone tracks
      try { lastAIStream.getTracks().forEach(t => t.stop()); } catch(e){}
      isAIRecording = false;
      aiRecordBtn.disabled = false;
      aiStopBtn.disabled = true;
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
}

aiRecordBtn.addEventListener("click", async () => {
  if (isAIRecording) return;
  await startAIRecording();
});

aiStopBtn.addEventListener("click", () => {
  if (!isAIRecording || !aiMediaRecorder) return;
  aiMediaRecorder.stop();
  // mic will be released in onstop handler
  aiStatusText.innerText = "⏳ Processing your question...";
});

// Optional: auto-open conversation panel if session exists
if (SESSION_ID) {
  console.log('Session ID:', SESSION_ID);
}
