// static/app.js

const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const uploadBtn = document.getElementById('upload-btn');
const uploadInput = document.getElementById('upload-input');
const resultDiv = document.getElementById('result');
const audioPlayback = document.getElementById('audio-playback');

let mediaRecorder;
let audioChunks = [];

/* ✅ NEW: Ask for microphone access when page loads */
window.addEventListener('load', async () => {
  try {
    await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log("🎤 Microphone access granted.");
  } catch (err) {
    alert("⚠️ Please allow microphone access in your browser to use the recording feature.");
    console.error("Microphone access denied:", err);
  }
});
/* ✅ END NEW CODE */

recordBtn.onclick = async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
    audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      audioPlayback.src = URL.createObjectURL(audioBlob);
      audioPlayback.play();
      await sendAudioToServer(audioBlob);
    };

    mediaRecorder.start();
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    resultDiv.innerHTML = "<p>🎙 Recording...</p>";
  } catch (err) {
    resultDiv.innerHTML = `<p style="color:red;">🎙 Microphone access denied or not available.</p>`;
    console.error(err);
  }
};

stopBtn.onclick = () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    resultDiv.innerHTML = "<p>Processing audio...</p>";
  }
};

// Handle manual file upload
uploadBtn.onclick = () => uploadInput.click();
uploadInput.onchange = async () => {
  const file = uploadInput.files[0];
  if (file) await sendAudioToServer(file);
};

async function sendAudioToServer(audioFile) {
  const formData = new FormData();
  formData.append('file', audioFile, 'recording.webm');

  try {
    const response = await fetch('/predict', { method: 'POST', body: formData });
    const data = await response.json();

    if (data.error) {
      resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
    } else {
      resultDiv.innerHTML = `
        <h3>Detected Animal:</h3>
        <p><b>${data.label}</b></p>
        <pre>${JSON.stringify(data.confidences, null, 2)}</pre>
      `;
    }
  } catch (error) {
    resultDiv.innerHTML = `<p style="color:red;">${error.message}</p>`;
  }
}
