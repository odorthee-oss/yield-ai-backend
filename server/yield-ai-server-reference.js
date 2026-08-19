/**
 * YIELD AI Backend
 * ----------------
 * Secure Claude-powered agricultural intelligence API.
 *
 * Required environment variables:
 *   ANTHROPIC_API_KEY
 *   YIELD_FRONTEND_ORIGIN
 *
 * Optional:
 *   PORT
 *   YIELD_MODEL
 */

const express = require("express");
const cors = require("cors");
const Anthropic = require("@anthropic-ai/sdk");

const app = express();

const PORT = Number(process.env.PORT) || 3000;
const MODEL = process.env.YIELD_MODEL || "claude-sonnet-5";
const FRONTEND_ORIGIN = process.env.YIELD_FRONTEND_ORIGIN;

if (!process.env.ANTHROPIC_API_KEY) {
  console.warn("YIELD WARNING: ANTHROPIC_API_KEY is not configured.");
}

/* ---------------------------------------------------------------
   CORS
---------------------------------------------------------------- */

const LOCAL_ORIGINS = new Set([
  "http://localhost:3000",
  "http://localhost:5173",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:5173",
]);

const corsOptions = {
  origin(origin, callback) {
    // Allow non-browser requests such as health checks and curl.
    if (!origin) {
      return callback(null, true);
    }

    if (FRONTEND_ORIGIN && origin === FRONTEND_ORIGIN) {
      return callback(null, true);
    }

    if (!FRONTEND_ORIGIN && LOCAL_ORIGINS.has(origin)) {
      return callback(null, true);
    }

    return callback(new Error("Not allowed by CORS"));
  },
  credentials: false,
};

app.use(cors(corsOptions));

/* ---------------------------------------------------------------
   BODY PARSING
---------------------------------------------------------------- */

app.use(express.json({ limit: "20kb" }));

/* ---------------------------------------------------------------
   ANTHROPIC CLIENT
---------------------------------------------------------------- */

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

/* ---------------------------------------------------------------
   YIELD SYSTEM PROMPT
---------------------------------------------------------------- */

const YIELD_SYSTEM_PROMPT = `
You are YIELD AI, an agricultural decision-support assistant for
smallholder farmers in Nigeria, currently supporting maize.

You are given the farmer's context and their question.

Use the farmer's actual context to personalize every answer.

FARM CONTEXT MAY INCLUDE:
- farmer name
- location
- farm size
- crop
- planting date
- crop age
- growth stage

RULES:

1. Ground your answer in the farmer's actual context.

2. Consider how the crop's current growth stage changes what matters.

3. Give practical, plain-language advice that a farmer can understand
   and act on immediately.

4. If there is not enough information to provide useful advice,
   clearly say so and ask a specific clarifying question.

5. Never present a pest, disease, or other agricultural diagnosis as
   confirmed based only on a text description. Describe possible causes
   and explain what would increase confidence.

6. Never invent current weather, market prices, soil test results,
   pest prevalence, rainfall, or other data that was not supplied.

7. Never recommend a specific chemical, pesticide, herbicide,
   fungicide, or dosage that could be unsafe or unapproved.

8. When a situation is serious, rapidly worsening, or uncertain,
   recommend consultation with a qualified local agricultural extension
   officer.

9. Prioritize the single most useful next action.

10. Keep the language warm, simple, practical, and respectful.

11. Do not pretend to have access to a live weather service, satellite
    imagery, laboratory results, or crop photographs unless those are
    actually provided.

12. The farmer is using YIELD as a decision-support assistant, not as
    a replacement for professional agricultural advice.

For insufficient-information responses:
- Set insufficient to true.
- Provide a short, specific clarifying question in text.

For useful responses:
- Set insufficient to false.
- Provide:
  whatMayBeHappening
  whatToCheck
  whatToDo
  recommendation
  note

The recommendation should be the most important action the farmer
should take next.
`.trim();

/* ---------------------------------------------------------------
   STRUCTURED OUTPUT SCHEMA
---------------------------------------------------------------- */

const YIELD_RESPONSE_SCHEMA = {
  type: "object",
  additionalProperties: false,
  properties: {
    insufficient: {
      type: "boolean",
    },

    text: {
      type: ["string", "null"],
    },

    whatMayBeHappening: {
      type: ["string", "null"],
    },

    whatToCheck: {
      type: ["array", "null"],
      items: {
        type: "string",
      },
    },

    whatToDo: {
      type: ["array", "null"],
      items: {
        type: "string",
      },
    },

    recommendation: {
      type: ["string", "null"],
    },

    note: {
      type: ["string", "null"],
    },
  },

  required: [
    "insufficient",
    "text",
    "whatMayBeHappening",
    "whatToCheck",
    "whatToDo",
    "recommendation",
    "note",
  ],
};

/* ---------------------------------------------------------------
   INPUT VALIDATION
---------------------------------------------------------------- */

function isNonEmptyString(value, maxLength) {
  return (
    typeof value === "string" &&
    value.trim().length > 0 &&
    value.length <= maxLength
  );
}

function validateFarmContext(context) {
  if (!context || typeof context !== "object") {
    return "farmContext must be an object";
  }

  const requiredFields = [
    "farmerName",
    "location",
    "farmSize",
    "crop",
    "plantingDate",
    "growthStage",
  ];

  for (const field of requiredFields) {
    if (!isNonEmptyString(context[field], 300)) {
      return `Invalid farmContext field: ${field}`;
    }
  }

  if (
    context.cropAgeDays !== undefined &&
    (
      typeof context.cropAgeDays !== "number" ||
      !Number.isFinite(context.cropAgeDays) ||
      context.cropAgeDays < 0 ||
      context.cropAgeDays > 10000
    )
  ) {
    return "cropAgeDays must be a valid non-negative number";
  }

  return null;
}

/* ---------------------------------------------------------------
   RESPONSE VALIDATION
---------------------------------------------------------------- */

function validateYieldResponse(response) {
  if (!response || typeof response !== "object") {
    return null;
  }

  if (typeof response.insufficient !== "boolean") {
    return null;
  }

  if (response.insufficient === true) {
    if (!isNonEmptyString(response.text, 1000)) {
      return null;
    }

    return {
      insufficient: true,
      text: response.text.trim(),
    };
  }

  if (!isNonEmptyString(response.whatMayBeHappening, 2000)) {
    return null;
  }

  if (
    !Array.isArray(response.whatToCheck) ||
    response.whatToCheck.length === 0 ||
    response.whatToCheck.some(
      (item) => !isNonEmptyString(item, 500)
    )
  ) {
    return null;
  }

  if (
    !Array.isArray(response.whatToDo) ||
    response.whatToDo.length === 0 ||
    response.whatToDo.some(
      (item) => !isNonEmptyString(item, 500)
    )
  ) {
    return null;
  }

  if (!isNonEmptyString(response.recommendation, 1500)) {
    return null;
  }

  if (!isNonEmptyString(response.note, 1500)) {
    return null;
  }

  return {
    insufficient: false,
    whatMayBeHappening: response.whatMayBeHappening.trim(),
    whatToCheck: response.whatToCheck.map((x) => x.trim()),
    whatToDo: response.whatToDo.map((x) => x.trim()),
    recommendation: response.recommendation.trim(),
    note: response.note.trim(),
  };
}

/* ---------------------------------------------------------------
   USER PROMPT
---------------------------------------------------------------- */

function buildUserPrompt(question, farmContext) {
  return `
FARMER CONTEXT

Name: ${farmContext.farmerName}
Location: ${farmContext.location}
Farm size: ${farmContext.farmSize}
Crop: ${farmContext.crop}
Planting date: ${farmContext.plantingDate}
Crop age: ${farmContext.cropAgeDays ?? "unknown"} days
Growth stage: ${farmContext.growthStage}

FARMER QUESTION

${question.trim()}
`.trim();
}

/* ---------------------------------------------------------------
   HEALTH CHECK
---------------------------------------------------------------- */

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    service: "yield-ai-backend",
    model: MODEL,
    apiKeyConfigured: Boolean(process.env.ANTHROPIC_API_KEY),
    corsConfigured: Boolean(FRONTEND_ORIGIN),
  });
});

/* ---------------------------------------------------------------
   AI ENDPOINT
---------------------------------------------------------------- */

app.post("/api/yield-ai", async (req, res) => {
  try {
    const { question, farmContext } = req.body || {};

    if (
      typeof question !== "string" ||
      question.trim().length === 0
    ) {
      return res.status(400).json({
        error: "A question is required.",
      });
    }

    if (question.length > 1000) {
      return res.status(400).json({
        error: "Question is too long.",
      });
    }

    const contextError = validateFarmContext(farmContext);

    if (contextError) {
      return res.status(400).json({
        error: contextError,
      });
    }

    if (!process.env.ANTHROPIC_API_KEY) {
      return res.status(503).json({
        error: "AI service is not configured.",
      });
    }

    const message = await anthropic.messages.create({
      model: MODEL,
      max_tokens: 1200,

      system: YIELD_SYSTEM_PROMPT,

      messages: [
        {
          role: "user",
          content: buildUserPrompt(question, farmContext),
        },
      ],

      output_config: {
        format: {
          type: "json_schema",
          schema: YIELD_RESPONSE_SCHEMA,
        },
      },
    });

    if (message.stop_reason === "refusal") {
      console.error("YIELD AI refused the request.");

      return res.status(502).json({
        error: "AI could not provide a response.",
      });
    }

    if (message.stop_reason === "max_tokens") {
      console.error("YIELD AI response reached max_tokens.");

      return res.status(502).json({
        error: "AI response was incomplete.",
      });
    }

    const textBlock = message.content?.find(
      (block) => block.type === "text"
    );

    if (!textBlock || typeof textBlock.text !== "string") {
      console.error("YIELD AI returned no usable text.");

      return res.status(502).json({
        error: "AI returned an unusable response.",
      });
    }

    let parsed;

    try {
      parsed = JSON.parse(textBlock.text);
    } catch (error) {
      console.error(
        "YIELD AI returned invalid structured data."
      );

      return res.status(502).json({
        error: "AI response could not be processed.",
      });
    }

    const validated = validateYieldResponse(parsed);

    if (!validated) {
      console.error(
        "YIELD AI response failed application validation."
      );

      return res.status(502).json({
        error: "AI response failed validation.",
      });
    }

    return res.json(validated);
  } catch (error) {
    console.error("YIELD AI backend error:", error);

    return res.status(500).json({
      error: "Something went wrong reaching the AI.",
    });
  }
});

/* ---------------------------------------------------------------
   ERROR HANDLER
---------------------------------------------------------------- */

app.use((error, req, res, next) => {
  if (error && error.message === "Not allowed by CORS") {
    return res.status(403).json({
      error: "Request origin is not allowed.",
    });
  }

  if (error && error.type === "entity.too.large") {
    return res.status(413).json({
      error: "Request is too large.",
    });
  }

  console.error("YIELD server error:", error);

  return res.status(500).json({
    error: "Internal server error.",
  });
});

/* ---------------------------------------------------------------
   START SERVER
---------------------------------------------------------------- */

app.listen(PORT, () => {
  console.log(
    `YIELD AI backend listening on port ${PORT}`
  );
});

module.exports = app;
