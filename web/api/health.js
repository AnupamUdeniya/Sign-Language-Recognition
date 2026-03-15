const { getHealth } = require("./_lib/hf-client");

async function handler(req, res) {
  if (req.method !== "GET") {
    res.setHeader("Allow", "GET");
    return res.status(405).json({ error: "Method not allowed." });
  }

  try {
    const payload = await getHealth();
    return res.status(200).json(payload);
  } catch (error) {
    return res.status(500).json({
      status: "error",
      error: error.message || "Health check failed."
    });
  }
}

module.exports = handler;
