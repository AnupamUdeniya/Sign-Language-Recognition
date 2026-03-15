const { predictFromDataUrl } = require("./_lib/hf-client");

async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed." });
  }

  try {
    const body = typeof req.body === "string"
      ? JSON.parse(req.body || "{}")
      : (req.body || {});

    const image = body.image || body.data_url;
    if (typeof image !== "string" || !image.startsWith("data:")) {
      return res.status(400).json({
        error: "Send a base64 image data URL in the `image` field."
      });
    }

    const result = await predictFromDataUrl(
      image,
      body.name || "capture.jpg",
      body.mimeType || "image/jpeg"
    );

    return res.status(200).json({
      ...result,
      backend: "huggingface-proxy"
    });
  } catch (error) {
    return res.status(500).json({
      error: error.message || "Prediction failed."
    });
  }
}

module.exports = handler;
module.exports.config = {
  api: {
    bodyParser: {
      sizeLimit: "8mb"
    }
  }
};
