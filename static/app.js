// 🎤 Enhanced frontend with permission handling, visualization & status tracking

const recordBtn = document.getElementById('record-btn');
const stopBtn = document.getElementById('stop-btn');
const uploadBtn = document.getElementById('upload-btn');
const uploadInput = document.getElementById('upload-input');
const resultDiv = document.getElementById('result');
const audioPlayback = document.getElementById('audio-playback');
const visualizer = document.getElementById('visualizer');
const statusIndicator = document.getElementById('status-indicator');
const historyList = document.getElementById('history-list');

let mediaRecorder;
let audioChunks = [];
let audioContext;
let analyser;
let animationFrame;
let detectionHistory = [];

// 🧩 Setup visualizer
function setupAudioVisualization() {
    if (!visualizer) return;
    visualizer.innerHTML = '';
    for (let i = 0; i < 32; i++) {
        const bar = document.createElement('div');
        bar.className = 'visualizer-bar';
        bar.style.setProperty('--bar-index', i);
        visualizer.appendChild(bar);
    }
}

// 🟢 Status indicator
function updateStatus(status, isError = false) {
    const dot = statusIndicator.querySelector('.status-dot');
    const text = statusIndicator.querySelector('.status-text');

    dot.style.backgroundColor = isError
        ? '#ff4757'
        : status === 'Recording'
        ? '#2ecc71'
        : '#4a6ef5';

    text.textContent = status;
}

// 🧠 Detection history
function addToHistory(detection) {
    const timestamp = new Date().toLocaleTimeString();
    detectionHistory.unshift({ detection, timestamp });
    detectionHistory = detectionHistory.slice(0, 5);
    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    historyList.innerHTML = detectionHistory
        .map(
            ({ detection, timestamp }) => `
        <div class="history-item">
            <div class="history-details">
                <span class="history-animal">${detection.label}</span>
                <span class="history-time">${timestamp}</span>
            </div>
            <span class="history-confidence">${Math.round(
                detection.confidence * 100
            )}%</span>
        </div>`
        )
        .join('');
}

// 🌀 Initialize visualization
async function initializeAudioContext(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.85;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    visualizeAudio();
}

// 🎨 Draw waveform & frequency bars
function visualizeAudio() {
    if (!visualizer) return;
    const bars = visualizer.querySelectorAll('.visualizer-bar');
    const frequencyData = new Uint8Array(analyser.frequencyBinCount);

    function animate() {
        animationFrame = requestAnimationFrame(animate);
        analyser.getByteFrequencyData(frequencyData);
        for (let i = 0; i < bars.length; i++) {
            const idx = Math.floor((i * frequencyData.length) / bars.length);
            const value = frequencyData[idx] / 255;
            const bar = bars[i];
            bar.style.height = `${value * 100}%`;
            const hue = 220 + value * 40;
            bar.style.backgroundColor = `hsla(${hue}, 80%, 60%, 0.8)`;
        }
    }
    animate();
}

// 🧩 Auto-request microphone permission
async function requestMicrophoneAccess() {
    try {
        await navigator.mediaDevices.getUserMedia({ audio: true });
        console.log("🎤 Microphone permission granted.");
        updateStatus('Ready');
        recordBtn.disabled = false;
    } catch (err) {
        console.warn("🚫 Microphone access denied:", err);
        alert(
            "⚠️ Microphone access is blocked.\n\nPlease click the 🔒 lock icon in your browser’s address bar → Site Settings → Allow microphone access."
        );
        updateStatus('Microphone Access Denied', true);
        recordBtn.disabled = true;
    }
}

// 🔘 Record start
recordBtn.onclick = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        await initializeAudioContext(stream);
        setupAudioVisualization();

        mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);
        mediaRecorder.onstop = async () => {
            cancelAnimationFrame(animationFrame);
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            audioPlayback.src = URL.createObjectURL(audioBlob);
            await sendAudioToServer(audioBlob);
        };

        mediaRecorder.start();
        recordBtn.disabled = true;
        stopBtn.disabled = false;
        updateStatus('Recording');
        resultDiv.innerHTML = '<p>🎙 Recording in progress...</p>';
    } catch (err) {
        console.error("Recording error:", err);
        updateStatus('Error', true);
        resultDiv.innerHTML = `<p style="color:red;">Microphone access denied or not available</p>`;
    }
};

// 🔴 Stop recording
stopBtn.onclick = () => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.disabled = false;
        stopBtn.disabled = true;
        updateStatus('Processing');
        resultDiv.innerHTML = '<p>Processing audio...</p>';
    }
};

// 📤 Handle file upload
uploadBtn.onclick = () => uploadInput.click();
uploadInput.onchange = async () => {
    const file = uploadInput.files[0];
    if (file) {
        updateStatus('Processing');
        await sendAudioToServer(file);
    }
};

// 🧠 Send to Flask backend
async function sendAudioToServer(audioFile) {
    const formData = new FormData();
    formData.append('file', audioFile);

    try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        const topPrediction = {
            label: data.label,
            confidence: Math.max(...Object.values(data.confidences))
        };

        resultDiv.innerHTML = `
            <div class="detection-result">
                <h3>Detected Animal</h3>
                <div class="prediction-main">
                    <span class="material-icons">pets</span>
                    <span class="animal-name">${topPrediction.label}</span>
                    <span class="confidence">${Math.round(topPrediction.confidence * 100)}%</span>
                </div>
                <div class="other-predictions">
                    ${Object.entries(data.confidences)
                        .sort(([,a],[,b]) => b - a)
                        .slice(1, 4)
                        .map(([label, conf]) => `
                            <div class="prediction-item">
                                <span class="label">${label}</span>
                                <span class="conf">${Math.round(conf * 100)}%</span>
                            </div>
                        `).join('')}
                </div>
            </div>
        `;
        addToHistory(topPrediction);
        updateStatus('Ready');
    } catch (error) {
        resultDiv.innerHTML = `<p style="color:red;">${error.message}</p>`;
        updateStatus('Error', true);
    }
}

// 🚀 Initialize app on page load
window.addEventListener('load', () => {
    setupAudioVisualization();
    requestMicrophoneAccess();
});
