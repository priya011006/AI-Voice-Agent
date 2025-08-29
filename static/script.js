// ================= Modern AI Voice Agent =================
let mediaRecorder;
let isRecording = false;
let isProcessing = false;
let lastStream = null;
let conversationHistory = [];

// DOM Elements
const mainRecordBtn = document.getElementById("main-record-btn");
const buttonIcon = document.getElementById("button-icon");
const statusText = document.getElementById("status-text");
const aiAudio = document.getElementById("ai-audio");
const conversationContainer = document.getElementById("conversation-container");
const sessionDisplay = document.getElementById("session-display");

// Button States
const ButtonState = {
  READY: 'ready',
  RECORDING: 'recording', 
  PROCESSING: 'processing',
  DISABLED: 'disabled'
};

// Session Management
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

// Update button appearance based on state
function updateButtonState(state) {
  mainRecordBtn.className = 'record-button';
  
  switch (state) {
    case ButtonState.READY:
      buttonIcon.textContent = "🎤";
      mainRecordBtn.disabled = false;
      statusText.className = "status-text";
      statusText.textContent = "Click to start recording your question";
      break;
      
    case ButtonState.RECORDING:
      buttonIcon.textContent = "⏸️";
      mainRecordBtn.classList.add('recording');
      mainRecordBtn.disabled = false;
      statusText.className = "status-text";
      statusText.textContent = "🎙️ Recording... Click to stop";
      break;
      
    case ButtonState.PROCESSING:
      buttonIcon.textContent = "⏳";
      mainRecordBtn.classList.add('processing');
      mainRecordBtn.disabled = true;
      statusText.className = "status-text";
      statusText.textContent = "🧠 Processing your question with AI...";
      break;
      
    case ButtonState.DISABLED:
      buttonIcon.textContent = "❌";
      mainRecordBtn.disabled = true;
      statusText.className = "status-text error";
      break;
  }
}

// Add conversation item to the display
function addConversationItem(type, content, isFallback = false) {
  const item = document.createElement('div');
  item.className = `conversation-item ${type}`;
  
  const label = document.createElement('div');
  label.className = 'conversation-label';
  label.textContent = type === 'user' ? 'You said' : 'AI responded';
  
  const contentDiv = document.createElement('div');
  contentDiv.className = 'conversation-content';
  contentDiv.textContent = content;
  
  item.appendChild(label);
  item.appendChild(contentDiv);
  
  if (isFallback) {
    const fallbackDiv = document.createElement('div');
    fallbackDiv.className = 'fallback-indicator';
    fallbackDiv.textContent = '(Technical issues detected - using fallback)';
    item.appendChild(fallbackDiv);
  }
  
  conversationContainer.appendChild(item);
  conversationContainer.style.display = 'block';
  
  // Scroll to bottom
  setTimeout(() => {
    item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

// Start recording
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    lastStream = stream;
    mediaRecorder = new MediaRecorder(stream);
    let audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = async () => {
      try {
        // Release microphone
        stream.getTracks().forEach(t => t.stop());
        
        updateButtonState(ButtonState.PROCESSING);
        
        const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
        const file = new File([audioBlob], "question.webm", { type: "audio/webm" });
        
        await processAudioWithAI(file);
        
      } catch (error) {
        console.error("Processing error:", error);
        handleError(error);
      }
    };

    mediaRecorder.start();
    isRecording = true;
    updateButtonState(ButtonState.RECORDING);
    
  } catch (err) {
    console.error("Recording Error:", err);
    handleMicrophoneError(err);
  }
}

// Stop recording
function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
  }
}

// Process audio with AI
async function processAudioWithAI(file) {
  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`/agent/chat/${SESSION_ID}`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.error || "AI conversation failed");
    }

    const result = await response.json();
    
    // Add user question to conversation
    const userText = result.transcript || 'Could not transcribe audio';
    addConversationItem('user', userText);
    
    // Add AI response to conversation
    const aiText = result.response || 'No response received';
    const isFallback = result.fallback_used || result.warning;
    addConversationItem('assistant', aiText, isFallback);
    
    // Update status based on result
    if (result.fallback_used) {
      statusText.className = "status-text warning";
      statusText.textContent = "⚠️ Responded with fallback due to technical issues";
    } else if (result.warning) {
      statusText.className = "status-text warning"; 
      statusText.textContent = "⚠️ Response generated with some technical issues";
    } else {
      statusText.className = "status-text success";
      statusText.textContent = "✅ Response complete! Ask another question";
    }
    
    // Play audio response
    if (result.audio_url) {
      await playAudioResponse(result.audio_url);
    }
    
    // Auto-start next recording after a brief delay
    setTimeout(() => {
      if (!isRecording && !isProcessing) {
        updateButtonState(ButtonState.READY);
        setTimeout(autoStartRecording, 1000);
      }
    }, 2000);
    
  } catch (error) {
    console.error("AI processing error:", error);
    handleError(error);
  }
}

// Play audio response
async function playAudioResponse(audioUrl) {
  return new Promise((resolve) => {
    aiAudio.src = audioUrl;
    
    aiAudio.oncanplay = () => {
      aiAudio.play().catch((error) => {
        console.error('Audio play failed:', error);
        statusText.className = "status-text warning";
        statusText.textContent = "⚠️ Audio generated but couldn't play automatically";
      });
    };
    
    aiAudio.onended = () => {
      resolve();
    };
    
    aiAudio.onerror = () => {
      console.error('Audio playback failed');
      statusText.className = "status-text warning";
      statusText.textContent = "⚠️ Audio response failed to play";
      resolve();
    };
    
    // Timeout fallback
    setTimeout(resolve, 10000);
  });
}

// Auto-start recording
function autoStartRecording() {
  if (!isRecording && !isProcessing) {
    startRecording();
  }
}

// Handle microphone errors
function handleMicrophoneError(error) {
  let message = "Microphone error occurred";
  
  if (error.name === 'NotAllowedError') {
    message = "⚠️ Microphone access denied. Please allow microphone permission and refresh.";
  } else if (error.name === 'NotFoundError') {
    message = "⚠️ No microphone detected. Please connect a microphone and refresh.";
  } else if (error.name === 'NotSupportedError') {
    message = "⚠️ Your browser doesn't support voice recording. Try Chrome, Firefox, or Safari.";
  }
  
  statusText.className = "status-text error";
  statusText.textContent = message;
  updateButtonState(ButtonState.DISABLED);
}

// Handle general errors
function handleError(error) {
  let message = "Something went wrong. Please try again.";
  
  if (error.message.includes('fetch') || error.message.includes('network')) {
    message = "❌ Network error. Check your connection and try again.";
  } else if (error.message.includes('audio')) {
    message = "❌ Audio processing failed. Please try again.";
  }
  
  statusText.className = "status-text error";
  statusText.textContent = message;
  
  // Reset to ready state after a delay
  setTimeout(() => {
    updateButtonState(ButtonState.READY);
  }, 3000);
}

// Main button click handler
mainRecordBtn.addEventListener("click", () => {
  if (isProcessing) return;
  
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Initialize UI
function initializeUI() {
  updateButtonState(ButtonState.READY);
  sessionDisplay.textContent = `Session: ${SESSION_ID}`;
  console.log('AI Voice Agent initialized with session:', SESSION_ID);
}

// Start the application
initializeUI();
