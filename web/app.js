const HF_API =
  "https://anupam090-asl-sign-language-recognition.hf.space/run/predict";

const classes = [
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N",
  "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
  "del", "nothing", "space"
];

const predictionLetter = document.getElementById("predictionLetter");
const predictionLabel = document.getElementById("predictionLabel");
const predictionConfidence = document.getElementById("predictionConfidence");
const phraseBox = document.getElementById("phraseBox");
const statusMessage = document.getElementById("statusMessage");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const liveDetectBtn = document.getElementById("liveDetectBtn");
const analyzeFrameBtn = document.getElementById("analyzeFrameBtn");

const imageUpload = document.getElementById("imageUpload");
const cameraFeed = document.getElementById("cameraFeed");
const captureCanvas = document.getElementById("captureCanvas");

let currentPrediction = { label: "nothing", confidence: 0 };
let mediaStream = null;
let liveDetectionTimer = null;
let requestInFlight = false;

function prettyLabel(label) {
  if (label === "del") return "Delete";
  if (label === "nothing") return "No sign";
  if (label === "space") return "Space";
  return `Letter ${label}`;
}

function normalizeLabel(label) {
  if (typeof label !== "string") return "nothing";

  const cleaned = label.trim();
  if (!cleaned) return "nothing";

  if (classes.includes(cleaned)) return cleaned;

  const upper = cleaned.toUpperCase();
  if (classes.includes(upper)) return upper;

  const lower = cleaned.toLowerCase();
  if (classes.includes(lower)) return lower;

  return "nothing";
}

function setStatus(msg) {
  if (statusMessage) {
    statusMessage.textContent = msg;
  }
}

function updatePrediction(result) {
  currentPrediction = result;

  if (predictionLetter) {
    predictionLetter.textContent =
      result.label === "space" ? "_" :
      result.label === "nothing" ? "?" :
      result.label;
  }

  if (predictionLabel) {
    predictionLabel.textContent = prettyLabel(result.label);
  }

  if (predictionConfidence) {
    predictionConfidence.textContent = `${result.confidence}%`;
  }
}

async function startCamera() {
  try {
    if (mediaStream) {
      setStatus("Camera already started");
      return;
    }

    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    });

    cameraFeed.srcObject = mediaStream;
    cameraFeed.playsInline = true;
    cameraFeed.muted = true;
    await cameraFeed.play();

    setStatus("Camera started");
  } catch (err) {
    setStatus("Camera error: " + err.message);
  }
}

function stopCamera() {
  if (liveDetectionTimer) {
    clearInterval(liveDetectionTimer);
    liveDetectionTimer = null;
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
  }

  mediaStream = null;
  cameraFeed.srcObject = null;
  setStatus("Camera stopped");
}

function captureCurrentFrameBlob() {
  return new Promise((resolve, reject) => {
    if (!cameraFeed.videoWidth || !cameraFeed.videoHeight) {
      reject(new Error("Camera not ready yet"));
      return;
    }

    captureCanvas.width = cameraFeed.videoWidth;
    captureCanvas.height = cameraFeed.videoHeight;

    const ctx = captureCanvas.getContext("2d");
    ctx.drawImage(cameraFeed, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(
      (blob) => {
        if (!blob) {
          reject(new Error("Unable to capture frame"));
          return;
        }
        resolve(blob);
      },
      "image/jpeg",
      0.92
    );
  });
}

function fileToDataURL(fileOrBlob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Failed to read image file"));

    reader.readAsDataURL(fileOrBlob);
  });
}

async function buildGradioImagePayload(fileOrBlob) {
  const dataUrl = await fileToDataURL(fileOrBlob);

  return {
    path: null,
    url: dataUrl,
    orig_name: fileOrBlob.name || "capture.jpg",
    size: fileOrBlob.size || 0,
    mime_type: fileOrBlob.type || "image/jpeg",
    is_stream: false,
    meta: { _type: "gradio.FileData" }
  };
}

function extractLabel(result) {
  let value = result?.data?.[0];

  if (Array.isArray(value)) {
    value = value[0];
  }

  if (typeof value === "string") {
    return normalizeLabel(value);
  }

  if (value && typeof value === "object") {
    if (typeof value.label === "string") return normalizeLabel(value.label);
    if (typeof value.text === "string") return normalizeLabel(value.text);
    if (typeof value.value === "string") return normalizeLabel(value.value);
  }

  return "nothing";
}

async function sendImageForPrediction(fileOrBlob) {
  if (requestInFlight) return;

  requestInFlight = true;
  setStatus("Running AI model...");

  try {
    const imgPayload = await buildGradioImagePayload(fileOrBlob);

    const response = await fetch(HF_API, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        data: [imgPayload]
      })
    });

    const rawText = await response.text();
    let result;

    try {
      result = JSON.parse(rawText);
    } catch {
      throw new Error("Invalid API response");
    }

    if (!response.ok) {
      throw new Error(result?.error || `HTTP ${response.status}`);
    }

    const label = extractLabel(result);

    updatePrediction({
      label,
      confidence: 100
    });

    setStatus(`Detected ${prettyLabel(label)}`);
  } catch (err) {
    console.error(err);
    updatePrediction({
      label: "nothing",
      confidence: 0
    });
    setStatus("Prediction error: " + err.message);
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
  } catch (err) {
    setStatus(err.message);
  }
}

function toggleLiveDetection() {
  if (!mediaStream) {
    setStatus("Start camera first");
    return;
  }

  if (liveDetectionTimer) {
    clearInterval(liveDetectionTimer);
    liveDetectionTimer = null;
    setStatus("Live detection stopped");
    return;
  }

  liveDetectionTimer = setInterval(() => {
    if (!requestInFlight) {
      analyzeCurrentFrame();
    }
  }, 1500);

  setStatus("Live detection running");
}

function handleUpload(e) {
  const file = e.target.files?.[0];
  if (!file) return;

  sendImageForPrediction(file);
  e.target.value = "";
}

startCameraBtn.addEventListener("click", startCamera);
stopCameraBtn.addEventListener("click", stopCamera);
liveDetectBtn.addEventListener("click", toggleLiveDetection);
analyzeFrameBtn.addEventListener("click", analyzeCurrentFrame);
imageUpload.addEventListener("change", handleUpload);

setStatus("Connected to HuggingFace model");
