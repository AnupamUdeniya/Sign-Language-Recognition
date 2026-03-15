const API_ENDPOINTS = {
  health: "/api/health",
  predict: "/api/predict"
};

const classes = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
  "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "del", "nothing", "space"
];

const predictionLetter = document.getElementById("predictionLetter");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const confidenceList = document.getElementById("confidenceList");
const phraseBox = document.getElementById("phraseBox");
const statusMessage = document.getElementById("statusMessage");
const classGrid = document.getElementById("classGrid");
const imagePreview = document.getElementById("imagePreview");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const liveDetectBtn = document.getElementById("liveDetectBtn");
const analyzeFrameBtn = document.getElementById("analyzeFrameBtn");
const clearPhraseBtn = document.getElementById("clearPhraseBtn");
const appendPredictionBtn = document.getElementById("appendPredictionBtn");
const addSpaceBtn = document.getElementById("addSpaceBtn");
const deleteCharBtn = document.getElementById("deleteCharBtn");
const checkBackendBtn = document.getElementById("checkBackendBtn");

const imageUpload = document.getElementById("imageUpload");
const cameraFeed = document.getElementById("cameraFeed");
const captureCanvas = document.getElementById("captureCanvas");

let currentPrediction = {
  label: "nothing",
  confidence: 0,
  top_predictions: []
};
let currentPhrase = "";
let mediaStream = null;
let liveDetectionTimer = null;
let requestInFlight = false;
let previewUrl = "";

function prettyLabel(label) {
  if (label === "del") return "Delete";
  if (label === "nothing") return "No sign";
  if (label === "space") return "Space";
  return `Letter ${label}`;
}

function normalizeLabel(label) {
  if (typeof label !== "string") return "nothing";

  const trimmed = label.trim();
  if (!trimmed) return "nothing";

  if (classes.includes(trimmed)) return trimmed;

  const upper = trimmed.toUpperCase();
  if (classes.includes(upper)) return upper;

  const lower = trimmed.toLowerCase();
  if (classes.includes(lower)) return lower;

  const tokenMatch = trimmed.match(/\b([A-Za-z]+)\b/);
  if (!tokenMatch) return "nothing";

  const token = tokenMatch[1];
  if (classes.includes(token)) return token;

  const tokenUpper = token.toUpperCase();
  if (classes.includes(tokenUpper)) return tokenUpper;

  const tokenLower = token.toLowerCase();
  if (classes.includes(tokenLower)) return tokenLower;

  return "nothing";
}

function normalizeConfidence(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const percentage = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Math.round(percentage * 100) / 100));
}

function parseTextPrediction(text) {
  const label = normalizeLabel(text);
  const confidenceMatch = text.match(/confidence:\s*([0-9.]+)/i);
  const confidence = confidenceMatch
    ? normalizeConfidence(confidenceMatch[1])
    : label === "nothing" ? 0 : 100;

  return {
    label,
    confidence,
    top_predictions: [
      {
        label,
        confidence
      }
    ]
  };
}

function normalizeTopPredictions(topPredictions, fallbackLabel, fallbackConfidence) {
  if (!Array.isArray(topPredictions) || topPredictions.length === 0) {
    return [
      {
        label: fallbackLabel,
        confidence: fallbackConfidence
      }
    ];
  }

  return topPredictions
    .map((item) => {
      const label = normalizeLabel(item?.label);
      const confidence = normalizeConfidence(item?.confidence);
      return { label, confidence };
    })
    .filter((item) => item.label !== "nothing" || item.confidence > 0)
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, 5);
}

function normalizePredictionPayload(payload) {
  if (!payload) {
    return {
      label: "nothing",
      confidence: 0,
      top_predictions: []
    };
  }

  if (typeof payload === "string") {
    return parseTextPrediction(payload);
  }

  if (Array.isArray(payload)) {
    return normalizePredictionPayload(payload[0]);
  }

  if (Array.isArray(payload.data)) {
    return normalizePredictionPayload(payload.data[0]);
  }

  if (typeof payload.label === "string") {
    const label = normalizeLabel(payload.label);
    const confidence = normalizeConfidence(payload.confidence);
    return {
      label,
      confidence,
      top_predictions: normalizeTopPredictions(
        payload.top_predictions,
        label,
        confidence
      )
    };
  }

  const labelScores = Object.entries(payload)
    .map(([label, score]) => ({
      label: normalizeLabel(label),
      confidence: normalizeConfidence(score)
    }))
    .filter((item) => item.label !== "nothing" || item.confidence > 0)
    .sort((left, right) => right.confidence - left.confidence)
    .slice(0, 5);

  if (labelScores.length > 0) {
    return {
      label: labelScores[0].label,
      confidence: labelScores[0].confidence,
      top_predictions: labelScores
    };
  }

  return {
    label: "nothing",
    confidence: 0,
    top_predictions: []
  };
}

function setStatus(message) {
  if (statusMessage) {
    statusMessage.textContent = message;
  }
}

function setFeedState(mode) {
  const showCamera = mode === "camera";
  const showImage = mode === "image";

  if (cameraFeed) {
    cameraFeed.hidden = !showCamera;
    cameraFeed.classList.toggle("active", showCamera);
  }

  if (imagePreview) {
    imagePreview.hidden = !showImage;
    imagePreview.classList.toggle("active", showImage);
  }

  if (cameraPlaceholder) {
    cameraPlaceholder.hidden = showCamera || showImage;
  }
}

function renderConfidenceList(predictions) {
  if (!confidenceList) return;

  const items = predictions.length > 0
    ? predictions
    : [{ label: "nothing", confidence: 0 }];

  confidenceList.innerHTML = "";

  items.forEach((item) => {
    const wrapper = document.createElement("div");
    wrapper.className = "confidence-item";

    const head = document.createElement("div");
    head.className = "confidence-head";
    head.innerHTML = `
      <span>${prettyLabel(item.label)}</span>
      <strong>${item.confidence}%</strong>
    `;

    const track = document.createElement("div");
    track.className = "confidence-track";

    const fill = document.createElement("div");
    fill.className = "confidence-fill";
    fill.style.width = `${item.confidence}%`;

    track.appendChild(fill);
    wrapper.appendChild(head);
    wrapper.appendChild(track);
    confidenceList.appendChild(wrapper);
  });
}

function renderPhrase() {
  if (!phraseBox) return;

  phraseBox.textContent = currentPhrase || "No phrase yet. Add predictions to build text.";
  phraseBox.classList.toggle("empty", currentPhrase.length === 0);
}

function highlightActiveClass(label) {
  document.querySelectorAll(".class-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.label === label);
  });
}

function updatePrediction(result) {
  const normalized = normalizePredictionPayload(result);
  currentPrediction = normalized;

  if (predictionLetter) {
    predictionLetter.textContent =
      normalized.label === "space" ? "_" :
      normalized.label === "nothing" ? "?" :
      normalized.label;
  }

  if (predictionLabel) {
    predictionLabel.textContent = prettyLabel(normalized.label);
  }

  if (predictionConfidence) {
    predictionConfidence.textContent = `${normalized.confidence}%`;
  }

  renderConfidenceList(normalized.top_predictions);
  highlightActiveClass(normalized.label);
}

function revokePreviewUrl() {
  if (!previewUrl) return;
  URL.revokeObjectURL(previewUrl);
  previewUrl = "";
}

function showUploadedPreview(file) {
  if (!imagePreview) return;

  revokePreviewUrl();
  previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  setFeedState("image");
}

function renderClassGrid() {
  if (!classGrid) return;

  classGrid.innerHTML = "";

  classes.forEach((label) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "class-chip";
    chip.dataset.label = label;
    chip.textContent = prettyLabel(label);
    chip.addEventListener("click", () => {
      appendLabelToPhrase(label);
      setStatus(`${prettyLabel(label)} added to phrase`);
    });
    classGrid.appendChild(chip);
  });
}

function appendLabelToPhrase(label) {
  const nextLabel = normalizeLabel(label);

  if (nextLabel === "nothing") {
    setStatus("No sign detected to add");
    return;
  }

  if (nextLabel === "space") {
    currentPhrase += " ";
  } else if (nextLabel === "del") {
    currentPhrase = currentPhrase.slice(0, -1);
  } else {
    currentPhrase += nextLabel;
  }

  renderPhrase();
}

function stopLiveDetection() {
  if (!liveDetectionTimer) return;

  clearInterval(liveDetectionTimer);
  liveDetectionTimer = null;

  if (liveDetectBtn) {
    liveDetectBtn.textContent = "Start live detection";
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus("This browser does not support camera access");
    return;
  }

  try {
    if (mediaStream) {
      setFeedState("camera");
      setStatus("Camera already started");
      return;
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user"
      },
      audio: false
    });

    cameraFeed.srcObject = mediaStream;
    cameraFeed.muted = true;
    cameraFeed.playsInline = true;
    await cameraFeed.play();

    setFeedState("camera");
    setStatus("Camera started. Capture a frame or enable live detection.");
  } catch (error) {
    const friendlyMessage = error?.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow access in the browser and try again."
      : `Camera error: ${error.message}`;
    setStatus(friendlyMessage);
  }
}

function stopCamera() {
  stopLiveDetection();

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  mediaStream = null;
  cameraFeed.srcObject = null;
  setFeedState(imagePreview?.src ? "image" : "idle");
  setStatus("Camera stopped");
}

function captureCurrentFrameBlob() {
  return new Promise((resolve, reject) => {
    if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
      reject(new Error("Camera not ready yet"));
      return;
    }

    const longestSide = Math.max(cameraFeed.videoWidth, cameraFeed.videoHeight);
    const scale = longestSide > 960 ? 960 / longestSide : 1;

    captureCanvas.width = Math.round(cameraFeed.videoWidth * scale);
    captureCanvas.height = Math.round(cameraFeed.videoHeight * scale);

    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Unable to capture frame"));
          return;
        }
        resolve(blob);
      },
      "image/jpeg",
      0.9
    );
  });
}

function fileToDataURL(fileOrBlob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read image"));

    reader.readAsDataURL(fileOrBlob);
  });
}

async function requestPrediction(fileOrBlob) {
  const image = await fileToDataURL(fileOrBlob);

  const response = await fetch(API_ENDPOINTS.predict, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      image,
      name: fileOrBlob.name || "capture.jpg",
      mimeType: fileOrBlob.type || "image/jpeg"
    })
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload?.error || `HTTP ${response.status}`);
  }

  return normalizePredictionPayload(payload);
}

async function sendImageForPrediction(fileOrBlob) {
  if (requestInFlight) return;

  requestInFlight = true;
  setStatus("Running AI model...");

  try {
    const result = await requestPrediction(fileOrBlob);
    updatePrediction(result);
    setStatus(
      `Detected ${prettyLabel(result.label)} with ${result.confidence}% confidence`
    );
  } catch (error) {
    console.error(error);
    updatePrediction({
      label: "nothing",
      confidence: 0,
      top_predictions: []
    });
    setStatus(`Prediction error: ${error.message}`);
  } finally {
    requestInFlight = false;
  }
}

async function analyzeCurrentFrame() {
  if (!mediaStream) {
    setStatus("Start camera first");
    return;
  }

  try {
    const blob = await captureCurrentFrameBlob();
    await sendImageForPrediction(blob);
  } catch (error) {
    setStatus(error.message);
  }
}

function toggleLiveDetection() {
  if (!mediaStream) {
    setStatus("Start camera first");
    return;
  }

  if (liveDetectionTimer) {
    stopLiveDetection();
    setStatus("Live detection stopped");
    return;
  }

  liveDetectionTimer = window.setInterval(() => {
    if (!requestInFlight) {
      analyzeCurrentFrame();
    }
  }, 1400);

  if (liveDetectBtn) {
    liveDetectBtn.textContent = "Stop live detection";
  }

  setStatus("Live detection running");
}

async function handleUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  showUploadedPreview(file);
  await sendImageForPrediction(file);
  event.target.value = "";
}

async function checkBackend(showReadyStatus = true) {
  try {
    const response = await fetch(API_ENDPOINTS.health);
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload?.error || `HTTP ${response.status}`);
    }

    if (showReadyStatus) {
      const backendLabel = payload?.backend || payload?.mode || "backend";
      setStatus(`Backend ready: ${backendLabel}`);
    }
  } catch (error) {
    setStatus(`Backend check failed: ${error.message}`);
  }
}

startCameraBtn?.addEventListener("click", startCamera);
stopCameraBtn?.addEventListener("click", stopCamera);
liveDetectBtn?.addEventListener("click", toggleLiveDetection);
analyzeFrameBtn?.addEventListener("click", analyzeCurrentFrame);
imageUpload?.addEventListener("change", handleUpload);
appendPredictionBtn?.addEventListener("click", () => appendLabelToPhrase(currentPrediction.label));
addSpaceBtn?.addEventListener("click", () => appendLabelToPhrase("space"));
deleteCharBtn?.addEventListener("click", () => appendLabelToPhrase("del"));
clearPhraseBtn?.addEventListener("click", () => {
  currentPhrase = "";
  renderPhrase();
  setStatus("Phrase cleared");
});
checkBackendBtn?.addEventListener("click", () => checkBackend(true));

window.addEventListener("beforeunload", stopCamera);

renderClassGrid();
renderPhrase();
updatePrediction(currentPrediction);
setFeedState("idle");
setStatus("Checking backend...");
checkBackend(true);
