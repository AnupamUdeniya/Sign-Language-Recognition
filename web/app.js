const classes = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
  "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
];

const confidenceList = document.getElementById("confidenceList");
const classGrid = document.getElementById("classGrid");
const predictionLetter = document.getElementById("predictionLetter");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const phraseBox = document.getElementById("phraseBox");
const statusMessage = document.getElementById("statusMessage");
const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const liveDetectBtn = document.getElementById("liveDetectBtn");
const analyzeFrameBtn = document.getElementById("analyzeFrameBtn");
const appendPredictionBtn = document.getElementById("appendPredictionBtn");
const addSpaceBtn = document.getElementById("addSpaceBtn");
const deleteCharBtn = document.getElementById("deleteCharBtn");
const clearPhraseBtn = document.getElementById("clearPhraseBtn");
const imageUpload = document.getElementById("imageUpload");
const cameraFeed = document.getElementById("cameraFeed");
const imagePreview = document.getElementById("imagePreview");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const captureCanvas = document.getElementById("captureCanvas");
const checkBackendBtn = document.getElementById("checkBackendBtn");

let currentPrediction = { label: "nothing", confidence: 0, top_predictions: [] };
let mediaStream = null;
let liveDetectionTimer = null;
let requestInFlight = false;

function prettyLabel(label) {
  if (label === "del") return "Delete";
  if (label === "nothing") return "No sign";
  if (label === "space") return "Space";
  return `Letter ${label}`;
}

function setStatus(message) {
  statusMessage.textContent = message;
}

function createConfidenceRows() {
  confidenceList.innerHTML = "";
  for (let index = 0; index < 5; index += 1) {
    const row = document.createElement("div");
    row.className = "confidence-item";
    row.innerHTML = `
      <div class="confidence-head">
        <span>Waiting...</span>
        <strong>0%</strong>
      </div>
      <div class="confidence-track">
        <div class="confidence-fill"></div>
      </div>
    `;
    confidenceList.appendChild(row);
  }
}

function updateConfidenceRows(predictions = []) {
  const rows = confidenceList.querySelectorAll(".confidence-item");

  rows.forEach((row, index) => {
    const fill = row.querySelector(".confidence-fill");
    const percent = row.querySelector("strong");
    const label = row.querySelector("span");
    const prediction = predictions[index];

    if (!prediction) {
      label.textContent = "-";
      percent.textContent = "0%";
      fill.style.width = "0%";
      return;
    }

    label.textContent = prettyLabel(prediction.label);
    percent.textContent = `${prediction.confidence}%`;
    fill.style.width = `${prediction.confidence}%`;
  });
}

function updatePrediction(result) {
  currentPrediction = result;
  const label = result.label;
  const confidence = result.confidence;

  predictionLetter.textContent = label === "space" ? "_" : label === "nothing" ? "?" : label;
  predictionLabel.textContent = prettyLabel(label);
  predictionConfidence.textContent = `${confidence}%`;

  document.querySelectorAll(".class-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.label === label);
  });

  updateConfidenceRows(result.top_predictions || []);
}

function buildClassGrid() {
  classes.forEach((label) => {
    const chip = document.createElement("button");
    chip.className = "class-chip";
    chip.type = "button";
    chip.dataset.label = label;
    chip.textContent = label;
    chip.addEventListener("click", () => {
      updatePrediction({
        label,
        confidence: 100,
        top_predictions: [{ label, confidence: 100 }]
      });
    });
    classGrid.appendChild(chip);
  });
}

function appendPrediction() {
  const currentText = phraseBox.textContent;

  if (currentPrediction.label === "space") {
    phraseBox.textContent = `${currentText} `;
    return;
  }

  if (currentPrediction.label === "del") {
    phraseBox.textContent = currentText.slice(0, -1);
    return;
  }

  if (currentPrediction.label === "nothing") {
    return;
  }

  phraseBox.textContent = `${currentText}${currentPrediction.label}`;
}

async function checkBackend() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("Health check failed.");
    }
    const data = await response.json();
    setStatus(`Backend ready on ${data.device}.`);
  } catch (error) {
    setStatus(`Backend unavailable: ${error.message}`);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    cameraPlaceholder.innerHTML = "<span>Camera unavailable</span><p>Your browser does not support webcam access here.</p>";
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    cameraFeed.srcObject = mediaStream;
    cameraFeed.hidden = false;
    cameraFeed.classList.add("active");
    imagePreview.hidden = true;
    imagePreview.classList.remove("active");
    cameraPlaceholder.hidden = true;
    cameraPlaceholder.style.display = "none";
    await cameraFeed.play();
    setStatus("Camera ready. Running first prediction...");
    setTimeout(() => {
      analyzeCurrentFrame();
    }, 500);
  } catch (error) {
    cameraPlaceholder.innerHTML = `<span>Access blocked</span><p>${error.message}</p>`;
    setStatus(`Camera error: ${error.message}`);
  }
}

function stopLiveDetection() {
  if (liveDetectionTimer) {
    clearInterval(liveDetectionTimer);
    liveDetectionTimer = null;
  }
  liveDetectBtn.textContent = "Start live detection";
}

function stopCamera() {
  stopLiveDetection();

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }

  cameraFeed.srcObject = null;
  cameraFeed.classList.remove("active");
  cameraPlaceholder.hidden = false;
  cameraPlaceholder.style.display = "grid";
  setStatus("Camera stopped.");
}

function dataUrlToBlob(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)[1];
  const bytes = atob(base64);
  const array = new Uint8Array(bytes.length);

  for (let i = 0; i < bytes.length; i += 1) {
    array[i] = bytes.charCodeAt(i);
  }

  return new Blob([array], { type: mime });
}

function captureCurrentFrameBlob() {
  if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
    throw new Error("Camera feed is not ready yet.");
  }

  captureCanvas.width = cameraFeed.videoWidth;
  captureCanvas.height = cameraFeed.videoHeight;
  const context = captureCanvas.getContext("2d");
  context.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.92);
  return dataUrlToBlob(dataUrl);
}

async function sendImageForPrediction(blob) {
  if (requestInFlight) {
    return;
  }

  requestInFlight = true;
  setStatus("Running model inference...");

  try {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    const response = await fetch("/predict", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    updatePrediction(data);
    setStatus(`Detected ${prettyLabel(data.label)} at ${data.confidence}%.`);
  } catch (error) {
    setStatus(`Prediction error: ${error.message}`);
  } finally {
    requestInFlight = false;
  }
}

async function analyzeCurrentFrame() {
  if (!mediaStream) {
    setStatus("Start the camera first.");
    return;
  }

  try {
    const blob = captureCurrentFrameBlob();
    await sendImageForPrediction(blob);
  } catch (error) {
    setStatus(`Capture error: ${error.message}`);
  }
}

function toggleLiveDetection() {
  if (liveDetectionTimer) {
    stopLiveDetection();
    setStatus("Live detection paused.");
    return;
  }

  if (!mediaStream) {
    setStatus("Start the camera first.");
    return;
  }

  liveDetectBtn.textContent = "Stop live detection";
  setStatus("Live detection started.");
  liveDetectionTimer = setInterval(() => {
    analyzeCurrentFrame();
  }, 1200);
}

function handleUpload(event) {
  const [file] = event.target.files;
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async () => {
    imagePreview.src = reader.result;
    imagePreview.hidden = false;
    imagePreview.classList.add("active");
    cameraFeed.classList.remove("active");
    cameraPlaceholder.hidden = true;
    stopCamera();
    await sendImageForPrediction(file);
  };
  reader.readAsDataURL(file);
}

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
liveDetectBtn.addEventListener("click", toggleLiveDetection);
analyzeFrameBtn.addEventListener("click", analyzeCurrentFrame);
appendPredictionBtn.addEventListener("click", appendPrediction);
addSpaceBtn.addEventListener("click", () => {
  phraseBox.textContent += " ";
});
deleteCharBtn.addEventListener("click", () => {
  phraseBox.textContent = phraseBox.textContent.slice(0, -1);
});
clearPhraseBtn.addEventListener("click", () => {
  phraseBox.textContent = "";
});
imageUpload.addEventListener("change", handleUpload);
checkBackendBtn.addEventListener("click", checkBackend);

createConfidenceRows();
buildClassGrid();
updatePrediction(currentPrediction);
checkBackend();


