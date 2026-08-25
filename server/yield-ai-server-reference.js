/**
 * YIELD AI — reference backend implementation (Node / Express)
 * ---------------------------------------------------------------
 * THIS FILE IS A REFERENCE, NOT WIRED INTO THE PROTOTYPE.
 *
 * It shows how /api/yield-ai should be implemented once YIELD has a
 * real server. The frontend (yield-app.jsx) already calls this route
 * via askYieldAI() and falls back to local mock logic if it's missing —
 * so deploying this file is what "turns on" live AI. No frontend
 * changes are needed.
 *
 * Uses the Google Gemini API (via the official @google/genai SDK) as
 * the model provider. The Gemini API key lives ONLY here, as a
 * server-side environment variable (GEMINI_API_KEY). It is never sent
 * to, or readable by, the browser.
 *
 * Install:  npm install
 * Run:      GEMINI_API_KEY=... node server/yield-ai-server-reference.js
 */

const express = require("express");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { Pool } = require("pg");
const { randomUUID } = require("crypto");
const { GoogleGenAI } = require("@google/genai");

const app = express();
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_URL
    ? { rejectUnauthorized: false }
    : false,
});

pool.on("error", (err) => {
  console.error("[yield-auth] Unexpected PostgreSQL pool error:", err);
});
/* ----------------------------------------------------------------
   CORS
   The frontend and backend are deployed on separate domains, so the
   browser needs an explicit CORS allowlist rather than same-origin
   defaults.

   YIELD_FRONTEND_ORIGIN should be set in production to the exact
   frontend origin (e.g. https://app.yield.farm — no trailing slash,
   no path). Only that origin is allowed to call this API.

   If YIELD_FRONTEND_ORIGIN is not set, we fall back to common local
   dev origins only (localhost/127.0.0.1 on typical dev ports) — never
   to "*". This means CORS will simply reject an unset production
   deploy rather than silently allow every origin, which is the safer
   failure mode.
------------------------------------------------------------------- */
const DEV_ORIGINS = [
  "http://localhost:3000",
  "http://localhost:5173",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:5173",
];

const ALLOWED_ORIGINS = process.env.YIELD_FRONTEND_ORIGIN
  ? [process.env.YIELD_FRONTEND_ORIGIN]
  : DEV_ORIGINS;

if (!process.env.YIELD_FRONTEND_ORIGIN) {
  console.warn(
    "[yield-ai] YIELD_FRONTEND_ORIGIN is not set — allowing local dev origins only " +
      `(${DEV_ORIGINS.join(", ")}). Set YIELD_FRONTEND_ORIGIN before deploying to production.`
  );
}

const corsOptions = {
  origin(origin, callback) {
    // Allow same-origin / non-browser requests (no Origin header, e.g.
    // curl or server-to-server health checks).
    if (!origin) return callback(null, true);
    if (ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
    return callback(new Error("Not allowed by CORS"));
  },
  methods: ["GET", "POST"],
  credentials: false, // no cookies/auth headers used by this API
};

app.use(cors(corsOptions));

// Reject oversized request bodies before they even reach our handler.
// 20kb is generous for a question + farm context; anything bigger is
// either a mistake or abuse.
app.use(express.json({ limit: "20kb" }));

const PORT = process.env.PORT || 3000;

const JWT_SECRET = process.env.JWT_SECRET;

if (!JWT_SECRET) {
  console.warn(
    "[yield-auth] JWT_SECRET is not set. Authentication endpoints will be unavailable."
  );
}

// Model used for YIELD AI. Overridable via env var so it can be bumped
// without a code change. gemini-3.5-flash-lite is a current, stable
// Gemini model positioned for cost-effective, high-throughput use and
// generous free-tier availability — a good fit for this MVP's simple
// structured-JSON task. Structured outputs (see below) are supported.

const MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";

if (!process.env.GEMINI_API_KEY) {
  // Don't crash on boot — this lets health checks pass during initial
  // deploy setup — but every real request will fail until it's set.
  console.warn("[yield-ai] GEMINI_API_KEY is not set. /api/yield-ai will return 502 until it is configured.");
}

const genAI = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY, // set on the server / hosting platform, never in frontend code
});

/* ----------------------------------------------------------------
   SYSTEM PROMPT
   Defines YIELD's role, tone, and hard safety boundaries. Sent with
   every request; never exposed to the browser. Unchanged from the
   original reference implementation.
------------------------------------------------------------------- */
const YIELD_SYSTEM_PROMPT = `
You are YIELD AI, an agricultural decision-support assistant for smallholder farmers in Nigeria, currently supporting maize.

You are given the farmer's context (name, location, farm size, crop, planting date, crop age, growth stage) and a question. Use that context to personalize your answer — reference the crop's actual growth stage and location where relevant.

Behavior rules:
1. Ground every answer in the farmer's actual context (crop, growth stage, location, farm size). Don't give generic advice that ignores it.
2. Consider how the crop's current growth stage changes what matters right now (e.g. nitrogen needs differ at vegetative vs. tasseling stages).
3. Give practical, plain-language advice a farmer with limited technical background can act on immediately. Avoid jargon; explain any technical term you must use.
4. If the question doesn't give you enough information to say anything useful, ask a specific clarifying question instead of guessing.
5. Clearly separate what you think MIGHT be happening from anything you'd call confirmed. Never state a pest or disease diagnosis as certain fact from a text description alone — describe it as a possibility, and say what would increase or reduce confidence (e.g. a clear photo).
6. NEVER invent specific data you don't have: don't state current weather, local pest prevalence, market prices, or soil test results as fact. If those matter, say so and suggest how the farmer could check.
7. NEVER recommend a specific chemical, pesticide, or dosage that could be unapproved or unsafe. Point to "a locally approved control method" or a local agricultural extension office instead of naming products or quantities.
8. When a situation sounds serious, uncertain, or worsening quickly, clearly recommend the farmer consult a qualified local agricultural extension officer — don't let your own advice stand in for that.
9. Prioritize the single most useful next action over a long list of general explanation.
10. Keep language simple, warm, and respectful — never condescending.

Respond according to the provided response schema. Use the "insufficient" variant only when you genuinely cannot give useful advice without more detail from the farmer. Otherwise use the full advice variant, and make sure every field is filled in with real, specific content grounded in the farmer's context — never leave a field empty or generic filler.
`.trim();

function buildUserPrompt(question, farmContext) {
  return `Farmer context:
- Name: ${farmContext.farmerName}
- Location: ${farmContext.location}
- Farm size: ${farmContext.farmSize}
- Crop: ${farmContext.crop}
- Planting date: ${farmContext.plantingDate}
- Crop age: ${farmContext.cropAgeDays ?? "unknown"} days
- Growth stage: ${farmContext.growthStage}

Farmer's question: "${question}"`;
}

/* ----------------------------------------------------------------
   STRUCTURED OUTPUT SCHEMA
   Passed to the Gemini API via config.responseSchema /
   config.responseMimeType: "application/json" (Gemini's native
   structured-outputs mechanism). This constrains Gemini's response at
   the model level via constrained decoding, so the model cannot
   return malformed JSON, wrap it in markdown, or add stray prose
   around it.

   We still deliberately do NOT trust this alone (see
   validateYieldResponse below) — structured outputs guarantee schema
   *shape* compliance, but Gemini can still stop early for safety
   reasons or hit the output token limit (finishReason other than
   "STOP"), in which case the returned text may not match the schema
   at all. Belt and braces for a farmer-facing MVP.

   Note: Gemini's schema format doesn't support "additionalProperties",
   so that constraint (present in the Anthropic version of this schema)
   is enforced only by validateYieldResponse below instead.
------------------------------------------------------------------- */
const YIELD_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    insufficient: { type: "boolean" },
    text: { type: "string" },
    whatMayBeHappening: { type: "string" },
    whatToCheck: { type: "array", items: { type: "string" } },
    whatToDo: { type: "array", items: { type: "string" } },
    recommendation: { type: "string" },
    note: { type: "string" },
  },
  required: ["insufficient"],
};

/* ----------------------------------------------------------------
   REQUEST VALIDATION
   Runs before we spend a single token calling Claude.
------------------------------------------------------------------- */
const MAX_QUESTION_LENGTH = 1000;
const MAX_CONTEXT_FIELD_LENGTH = 300;
const REQUIRED_CONTEXT_FIELDS = ["farmerName", "location", "farmSize", "crop", "plantingDate", "growthStage"];

function validateRequestBody(body) {
  if (!body || typeof body !== "object") return "Request body must be a JSON object.";

  const { question, farmContext } = body;

  if (typeof question !== "string" || !question.trim()) {
    return "question must be a non-empty string.";
  }
  if (question.length > MAX_QUESTION_LENGTH) {
    return `question must be under ${MAX_QUESTION_LENGTH} characters.`;
  }

  if (!farmContext || typeof farmContext !== "object" || Array.isArray(farmContext)) {
    return "farmContext is required and must be an object.";
  }

  for (const field of REQUIRED_CONTEXT_FIELDS) {
    const value = farmContext[field];
    if (typeof value !== "string" || !value.trim()) {
      return `farmContext.${field} is required.`;
    }
    if (value.length > MAX_CONTEXT_FIELD_LENGTH) {
      return `farmContext.${field} is too long.`;
    }
  }

  if (farmContext.cropAgeDays !== undefined) {
    const age = farmContext.cropAgeDays;
    if (typeof age !== "number" || !Number.isFinite(age) || age < 0 || age > 3650) {
      return "farmContext.cropAgeDays must be a non-negative number.";
    }
  }

  return null; // valid
}

/* ----------------------------------------------------------------
   RESPONSE VALIDATION
   Enforces the exact two-shape contract the frontend understands,
   independent of whatever the structured-output schema allowed
   through. Returns a clean, whitelisted object (only the fields the
   frontend expects) or null if the response doesn't satisfy either
   valid shape.
------------------------------------------------------------------- */
function isNonEmptyString(v) {
  return typeof v === "string" && v.trim().length > 0;
}

function isNonEmptyStringArray(v) {
  return Array.isArray(v) && v.length > 0 && v.every((item) => isNonEmptyString(item));
}

function validateYieldResponse(data) {
  if (!data || typeof data !== "object") return null;

  if (data.insufficient === true) {
    if (!isNonEmptyString(data.text)) return null;
    return { insufficient: true, text: data.text };
  }

  if (data.insufficient === false) {
    if (
      isNonEmptyString(data.whatMayBeHappening) &&
      isNonEmptyStringArray(data.whatToCheck) &&
      isNonEmptyStringArray(data.whatToDo) &&
      isNonEmptyString(data.recommendation) &&
      isNonEmptyString(data.note)
    ) {
      return {
        insufficient: false,
        whatMayBeHappening: data.whatMayBeHappening,
        whatToCheck: data.whatToCheck,
        whatToDo: data.whatToDo,
        recommendation: data.recommendation,
        note: data.note,
      };
    }
    return null;
  }

  return null; // insufficient missing or not a boolean
}

/* ----------------------------------------------------------------
   ROUTES
------------------------------------------------------------------- */

// Simple liveness check — useful for confirming a deploy went live
// before pointing the frontend at it.
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", provider: "gemini", model: MODEL, configured: Boolean(process.env.GEMINI_API_KEY) });
});

app.post("/api/yield-ai", async (req, res) => {
  const validationError = validateRequestBody(req.body);
  if (validationError) {
    return res.status(400).json({ error: validationError });
  }

  const { question, farmContext } = req.body;

  if (!process.env.GEMINI_API_KEY) {
    console.error("[yield-ai] Request received but GEMINI_API_KEY is not configured.");
    return res.status(502).json({ error: "The AI service is not configured yet. Please try again later." });
  }

  let response;
  try {
    response = await genAI.models.generateContent({
      model: MODEL,
      contents: buildUserPrompt(question, farmContext),
      config: {
        systemInstruction: YIELD_SYSTEM_PROMPT,
        maxOutputTokens: 1000,
        // Native structured outputs: constrains Gemini's output to this
        // schema via constrained decoding, instead of relying on
        // prompting alone + JSON.parse(raw).
        responseMimeType: "application/json",
        responseSchema: YIELD_RESPONSE_SCHEMA,
      },
    });
  } catch (err) {
    // Never leak err.message / stack to the client — log it server-side only.
    console.error("[yield-ai] Gemini API call failed:", err);
    return res.status(502).json({ error: "The AI service is temporarily unavailable. Please try again." });
  }

  const finishReason = response.candidates?.[0]?.finishReason;

  if (finishReason === "MAX_TOKENS") {
    console.warn("[yield-ai] Response was truncated at the token limit.");
    return res.status(502).json({ error: "The AI response was too long to complete. Please try a shorter or more specific question." });
  }

  if (finishReason && finishReason !== "STOP") {
    // SAFETY, RECITATION, PROHIBITED_CONTENT, OTHER, etc. — Gemini
    // stopped before producing a normal complete response.
    console.warn("[yield-ai] Model stopped for a non-STOP reason:", finishReason);
    return res.status(502).json({ error: "YIELD couldn't generate a response to that question. Please rephrase and try again." });
  }

  const rawText = response.text;
  if (!rawText) {
    console.error("[yield-ai] No text in model response:", JSON.stringify(response.candidates));
    return res.status(502).json({ error: "The AI service returned an unexpected response." });
  }

  let parsed;
  try {
    parsed = JSON.parse(rawText);
  } catch (parseErr) {
    // Should be rare with structured outputs enabled, but a truncation
    // or edge case can still produce non-JSON text — never forward that.
    console.error("[yield-ai] Failed to parse model output as JSON:", rawText);
    return res.status(502).json({ error: "The AI service returned malformed data." });
  }

  const validated = validateYieldResponse(parsed);
  if (!validated) {
    console.error("[yield-ai] Model output failed shape validation:", JSON.stringify(parsed));
    return res.status(502).json({ error: "The AI service returned an invalid response structure." });
  }

  return res.json(validated);
});

/* ----------------------------------------------------------------
   ERROR HANDLING
   Catches malformed JSON bodies, oversized payloads, and anything
   unexpected — always responds with a generic message, never the
   raw error, stack trace, or API key.
------------------------------------------------------------------- */
app.use((err, req, res, next) => {
  if (err && err.message === "Not allowed by CORS") {
    return res.status(403).json({ error: "This origin is not allowed to access the API." });
  }
  if (err && err.type === "entity.too.large") {
    return res.status(413).json({ error: "Request body is too large." });
  }
  if (err instanceof SyntaxError && err.status === 400 && "body" in err) {
    return res.status(400).json({ error: "Request body must be valid JSON." });
  }
  console.error("[yield-ai] Unexpected server error:", err);
  return res.status(500).json({ error: "Something went wrong. Please try again." });
});

app.listen(PORT, () => console.log(`[yield-ai] listening on port ${PORT} (model: ${MODEL})`));

module.exports = app;

