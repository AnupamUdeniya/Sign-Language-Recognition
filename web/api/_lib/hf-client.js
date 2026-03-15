const DEFAULT_SPACE_ID = "Anupam090/asl-sign-language-recognition";
const SPACE_ID =
  process.env.HUGGINGFACE_SPACE_ID ||
  process.env.HF_SPACE_ID ||
  DEFAULT_SPACE_ID;
const SPACE_URL = (
  process.env.HUGGINGFACE_SPACE_URL ||
  process.env.HF_SPACE_URL ||
  `https://${SPACE_ID.replace("/", "-").toLowerCase()}.hf.space`
).replace(/\/+$/, "");
const ENDPOINT_PATHS = ["/gradio_api/call/predict", "/call/predict"];

function getToken() {
  return (
    process.env.HUGGINGFACE_API_KEY ||
    process.env.HUGGINGFACE_TOKEN ||
    process.env.HF_TOKEN ||
    ""
  );
}

function getAuthHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

function normalizeLabel(label) {
  if (typeof label !== "string") return "nothing";

  const trimmed = label.trim();
  if (!trimmed) return "nothing";

  const match = trimmed.match(/\b([A-Za-z]+)\b/);
  if (!match) return "nothing";

  const value = match[1];
  if (value.length === 1) return value.toUpperCase();

  const lowered = value.toLowerCase();
  if (lowered === "del" || lowered === "nothing" || lowered === "space") {
    return lowered;
  }

  return "nothing";
}

function normalizeConfidence(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return 0;
  const percentage = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, Math.round(percentage * 100) / 100));
}

function buildFilePayload(dataUrl, fileName, mimeType) {
  const base64 = typeof dataUrl === "string" && dataUrl.includes(",")
    ? dataUrl.split(",", 1)[0]
    : "";
  const encoded = typeof dataUrl === "string" && dataUrl.includes(",")
    ? dataUrl.split(",", 2)[1]
    : "";

  return {
    path: null,
    url: dataUrl,
    size: Math.floor((encoded.length * 3) / 4),
    orig_name: fileName || "capture.jpg",
    mime_type: mimeType || base64.replace("data:", "").replace(";base64", "") || "image/jpeg",
    is_stream: false,
    meta: { _type: "gradio.FileData" }
  };
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
    top_predictions: [{ label, confidence }]
  };
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
    const topPredictions = Array.isArray(payload.top_predictions)
      ? payload.top_predictions.map((item) => ({
          label: normalizeLabel(item?.label),
          confidence: normalizeConfidence(item?.confidence)
        }))
      : [{ label, confidence }];

    return {
      label,
      confidence,
      top_predictions: topPredictions
        .filter((item) => item.label !== "nothing" || item.confidence > 0)
        .sort((left, right) => right.confidence - left.confidence)
        .slice(0, 5)
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

  if (labelScores.length === 0) {
    return {
      label: "nothing",
      confidence: 0,
      top_predictions: []
    };
  }

  return {
    label: labelScores[0].label,
    confidence: labelScores[0].confidence,
    top_predictions: labelScores
  };
}

async function safeJson(response) {
  const text = await response.text();
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

async function createPredictionJob(imagePayload) {
  let lastError = null;

  for (const endpointPath of ENDPOINT_PATHS) {
    const response = await fetch(`${SPACE_URL}${endpointPath}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...getAuthHeaders()
      },
      body: JSON.stringify({
        data: [imagePayload]
      })
    });

    const payload = await safeJson(response);

    if (response.status === 404) {
      lastError = new Error(`404 from ${endpointPath}`);
      continue;
    }

    if (!response.ok) {
      const message =
        payload?.error ||
        payload?.message ||
        (typeof payload === "string" ? payload : "") ||
        `Hugging Face returned HTTP ${response.status}`;
      throw new Error(message);
    }

    if (payload?.event_id) {
      return {
        endpointPath,
        eventId: payload.event_id
      };
    }

    if (payload?.data || typeof payload === "string" || Array.isArray(payload)) {
      return {
        endpointPath,
        immediate: payload
      };
    }

    throw new Error("Prediction endpoint returned no event_id.");
  }

  throw lastError || new Error("Prediction endpoint not found.");
}

function parseEventBlock(block) {
  const lines = block.split("\n");
  let eventName = "";
  let dataText = "";

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataText += line.slice(5).trim();
    }
  }

  return { eventName, dataText };
}

async function waitForPredictionResult(endpointPath, eventId) {
  const response = await fetch(`${SPACE_URL}${endpointPath}/${eventId}`, {
    method: "GET",
    headers: {
      Accept: "text/event-stream",
      ...getAuthHeaders()
    }
  });

  if (!response.ok) {
    const payload = await safeJson(response);
    const message =
      payload?.error ||
      payload?.message ||
      (typeof payload === "string" ? payload : "") ||
      `Hugging Face returned HTTP ${response.status}`;
    throw new Error(message);
  }

  if (!response.body) {
    throw new Error("Prediction stream did not return a response body.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";

    for (const block of blocks) {
      const { eventName, dataText } = parseEventBlock(block);
      if (!dataText || eventName === "heartbeat") {
        continue;
      }

      let parsed;
      try {
        parsed = JSON.parse(dataText);
      } catch {
        parsed = dataText;
      }

      if (eventName === "error") {
        const message =
          parsed?.error ||
          parsed?.message ||
          (typeof parsed === "string" ? parsed : "Prediction failed.");
        throw new Error(message);
      }

      if (eventName === "complete" || eventName === "data") {
        return parsed;
      }
    }

    if (done) {
      break;
    }
  }

  throw new Error("Prediction stream ended without a result.");
}

async function predictFromDataUrl(dataUrl, fileName, mimeType) {
  const imagePayload = buildFilePayload(dataUrl, fileName, mimeType);
  const job = await createPredictionJob(imagePayload);
  const rawResult =
    job.immediate ||
    await waitForPredictionResult(job.endpointPath, job.eventId);

  return normalizePredictionPayload(rawResult);
}

async function getHealth() {
  let lastError = null;

  for (const endpointPath of ["", ...ENDPOINT_PATHS]) {
    try {
      const response = await fetch(`${SPACE_URL}${endpointPath}`, {
        headers: {
          ...getAuthHeaders()
        }
      });

      if (response.ok || response.status === 405) {
        return {
          status: "ok",
          backend: "huggingface-proxy",
          spaceId: SPACE_ID,
          spaceUrl: SPACE_URL,
          hasToken: Boolean(getToken())
        };
      }

      lastError = new Error(`Health check returned HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError || new Error("Unable to reach the Hugging Face Space.");
}

module.exports = {
  getHealth,
  predictFromDataUrl
};
