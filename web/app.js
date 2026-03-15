const HF_SPACE = "https://anupam090-asl-sign-language-recognition.hf.space";
const HF_API = `${HF_SPACE}/call/predict`;

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

  const trimmed = label.trim();
  if (!trimmed) return "nothing";

  if (classes.includes(trimmed)) return trimmed;

  const upper = trimmed.toUpperCase();
  if (classes.includes(upper)) return upper;

  const lower = trimmed.toLowerCase();
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
    cameraFeed.muted = true;
    cameraFeed.playsInline = true;
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
    reader.onerror = () => reject(new Error("Failed to read image"));

    reader.readAsDataURL(fileOrBlob);
  });
}

async function buildImagePayload(fileOrBlob) {
  const dataUrl = await fileToDataURL(fileOrBlob);

  return {
    path: null,
    url: dataUrl,
    size: fileOrBlob.size || 0,
    orig_name: fileOrBlob.name || "capture.jpg",
    mime_type: fileOrBlob.type || "image/jpeg",
    is_stream: false,
    meta: { _type: "gradio.FileData" }
  };
}

function extractLabelFromCompletedData(parsedData) {
  if (Array.isArray(parsedData)) {
    return normalizeLabel(parsedData[0]);
  }

  if (typeof parsedData === "string") {
    return normalizeLabel(parsedData);
  }

  if (parsedData && Array.isArray(parsedData.data)) {
    return normalizeLabel(parsedData.data[0]);
  }

  return "nothing";
}

async function createPredictionJob(fileOrBlob) {
  const imagePayload = await buildImagePayload(fileOrBlob);

  const response = await fetch(HF_API, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      data: [imagePayload]
    })
  });

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(payload?.error || `HTTP ${response.status}`);
  }

  if (!payload?.event_id) {
    throw new Error("No event_id returned by API");
  }

  return payload.event_id;
}

async function waitForPredictionResult(eventId) {
  const response = await fetch(`${HF_API}/${eventId}`, {
    method: "GET",
    headers: {
      Accept: "text/event-stream"
    }
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error("No response stream from API");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const eventBlock of events) {
      const lines = eventBlock.split("\n");
      let eventName = "";
      let dataText = "";

      for (const line of lines) {
        if (line.startsWith("event:")) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith("data:")) {
          dataText += line.slice(5).trim();
        }
      }

      if (!dataText) continue;

      if (eventName === "heartbeat") continue;

      if (eventName === "error") {
        try {
          const parsedError = JSON.parse(dataText);
          throw new Error(parsedError?.error || "API returned an error");
        } catch {
          throw new Error("API returned an error");
        }
      }

      if (eventName === "complete") {
        let parsed;
        try {
          parsed = JSON.parse(dataText);
        } catch {
          throw new Error("Invalid prediction response");
        }

        return extractLabelFromCompletedData(parsed);
      }
    }

    if (done) break;
  }

  throw new Error("Prediction stream ended without a result");
}

async function sendImageForPrediction(fileOrBlob) {
  if (requestInFlight) return;

  requestInFlight = true;
  setStatus("Running AI model...");

  try {
    const eventId = await createPredictionJob(fileOrBlob);
    const label = await waitForPredictionResult(eventId);

    updatePrediction({
      label,
      confidence: label === "nothing" ? 0 : 100
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

startCameraBtn?.addEventListener("click", startCamera);
stopCameraBtn?.addEventListener("click", stopCamera);
liveDetectBtn?.addEventListener("click", toggleLiveDetection);
analyzeFrameBtn?.addEventListener("click", analyzeCurrentFrame);
imageUpload?.addEventListener("change", handleUpload);

setStatus("Connected to Hugging Face model");
