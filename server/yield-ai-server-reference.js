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
const helmet = require("helmet");
const { rateLimit } = require("express-rate-limit");
const { GoogleGenAI } = require("@google/genai");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const { Pool } = require("pg");
const { randomUUID } = require("crypto");

const app = express();

// Render sits in front of this app as a reverse proxy, so Express sees
// the proxy's IP on req.ip unless told to trust the proxy's
// X-Forwarded-For header. This is required both for correct client IPs
// in general and specifically for the auth rate limiter below to key
// on each real client rather than treating every request as coming
// from the same address. `1` trusts exactly one hop, matching Render's
// single reverse-proxy setup.
app.set("trust proxy", 1);

// Sensible-defaults security headers (X-Content-Type-Options, HSTS,
// X-Frame-Options, a default Content-Security-Policy, etc.). No custom
// CSP here — this is a pure JSON API with no HTML/script responses to
// protect, so Helmet's defaults are sufficient and low-risk.
app.use(helmet());

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
  methods: ["GET", "POST", "PUT"],
  credentials: false, // no cookies/auth headers used by this API
};

app.use(cors(corsOptions));

// Reject oversized request bodies before they even reach our handler.
// 20kb is generous for a question + farm context; anything bigger is
// either a mistake or abuse.
app.use(express.json({ limit: "20kb" }));

const PORT = process.env.PORT || 3000;

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
   AUTH / DATABASE SETUP (V1)
   Farm-name-based account identity, backed by Postgres. Nothing in
   this block is used by /api/yield-ai — the AI route has no
   dependency on the database or on authentication succeeding.
------------------------------------------------------------------- */
const JWT_SECRET = process.env.JWT_SECRET;

if (!JWT_SECRET) {
  // Don't crash on boot — /api/yield-ai and /api/health must keep
  // working even if auth isn't configured yet. Every /api/auth/* route
  // checks this itself and returns a generic 500 rather than exposing
  // the missing-secret detail or throwing.
  console.warn("[yield-ai] JWT_SECRET is not set. /api/auth/* will return a generic server error until it is configured.");
}

// DATABASE_URL is expected to be a standard Postgres connection string
// (e.g. Render's managed Postgres). SSL is enabled whenever a connection
// string is present, matching Render's requirements; never logged or
// exposed in any response.
const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_URL ? { rejectUnauthorized: false } : false,
});

let dbReady = false;

// Creates the farms table if it doesn't exist yet, then safely
// migrates it forward with any new columns using ADD COLUMN IF NOT
// EXISTS — this never touches or deletes existing rows/accounts, it
// only widens the table. Called once at startup (see bottom of file),
// without blocking app.listen — if the database is temporarily
// unavailable, we log it clearly and let /api/auth/* routes report a
// 503 per-request, rather than crashing the whole service or delaying
// the Gemini route from coming online.
async function initDatabase() {
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS farms (
        id UUID PRIMARY KEY,
        farm_name VARCHAR(150) UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT NOW()
      );
    `);

    // Farm profile fields (from onboarding / Edit Farm). Added via
    // ADD COLUMN IF NOT EXISTS so existing accounts are preserved as-is
    // — new columns simply default to NULL for rows that predate them,
    // and get filled in the next time the farmer saves their profile.
    await pool.query(`
      ALTER TABLE farms
        ADD COLUMN IF NOT EXISTS farmer_name VARCHAR(150),
        ADD COLUMN IF NOT EXISTS phone VARCHAR(30),
        ADD COLUMN IF NOT EXISTS location VARCHAR(300),
        ADD COLUMN IF NOT EXISTS farm_size VARCHAR(100),
        ADD COLUMN IF NOT EXISTS crop VARCHAR(100),
        ADD COLUMN IF NOT EXISTS planting_date DATE;
    `);

    // ---------------------------------------------------------------
    // V2 — users table (person/farmer identity), additive only.
    // "farms" is untouched except for a new nullable link column below.
    // No existing farm row is modified, deleted, or requires a user.
    // ---------------------------------------------------------------
    await pool.query(`
      CREATE TABLE IF NOT EXISTS users (
        id UUID PRIMARY KEY,
        full_name VARCHAR(150) NOT NULL,
        email VARCHAR(254) UNIQUE NOT NULL,
        phone VARCHAR(30),
        password_hash VARCHAR(255) NOT NULL,
        role VARCHAR(30) NOT NULL DEFAULT 'farmer',
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
      );
    `);

    // Defensive: covers the case where "users" already exists from a
    // partial prior deploy without "role" — a harmless no-op otherwise.
    await pool.query(`
      ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(30) NOT NULL DEFAULT 'farmer';
    `);

    // Link farms -> users. Nullable, and deliberately left nullable —
    // every farm row that exists today predates this relationship and
    // has no user to point to. They remain fully functional, legacy,
    // unlinked accounts (see GET/PUT /api/auth/me) until a future
    // migration explicitly links or prompts them. Never made NOT NULL.
    await pool.query(`
      ALTER TABLE farms ADD COLUMN IF NOT EXISTS user_id UUID;
    `);

    // Foreign key, added idempotently. Postgres has no
    // "ADD CONSTRAINT IF NOT EXISTS", so this checks pg_constraint
    // first — safe to run on every startup, including against a
    // database that already has this constraint from a prior run.
    await pool.query(`
      DO $$
      BEGIN
        IF NOT EXISTS (
          SELECT 1 FROM pg_constraint WHERE conname = 'farms_user_id_fkey'
        ) THEN
          ALTER TABLE farms
            ADD CONSTRAINT farms_user_id_fkey
            FOREIGN KEY (user_id) REFERENCES users(id);
        END IF;
      END $$;
    `);

    await pool.query(`
      CREATE INDEX IF NOT EXISTS idx_farms_user_id ON farms(user_id);
    `);

    dbReady = true;
    console.log("[yield-ai] Database ready — farms + users tables available (user_id link is additive/nullable).");
  } catch (err) {
    dbReady = false;
    console.error("[yield-ai] Database initialization failed — /api/auth/* will return 503 until this is resolved:", err.message);
  }
}

// Precomputed once at startup and used only as a stand-in hash when a
// login attempt targets an unknown farm name, so bcrypt.compare() still
// runs and login's response timing doesn't reveal whether the farm name
// exists. Not a real credential.
const DUMMY_PASSWORD_HASH = bcrypt.hashSync("yield-timing-safety-placeholder", 10);

const MAX_FARM_NAME_LENGTH = 150;
const MIN_PASSWORD_LENGTH = 8;
const MAX_PROFILE_FIELD_LENGTH = 300;

// V2 — users/identity layer additions.
const MAX_FULL_NAME_LENGTH = 150;
const MAX_EMAIL_LENGTH = 254;
const MAX_PHONE_LENGTH = 30;
const EMAIL_FORMAT_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

function normalizeEmail(value) {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

// Signs the app's JWT. Payload deliberately carries only IDs/name —
// never email, phone, or anything else sensitive. userId is null for
// legacy farm-only accounts with no linked user. Same secret and
// expiry as before (7d) — nothing about token lifetime changed.
function signAuthToken({ userId = null, farmId = null, farmName = null }) {
  return jwt.sign({ userId, farmId, farmName }, JWT_SECRET, { expiresIn: "7d" });
}

// Maps a `users` row to the public JSON shape. Deliberately excludes
// password_hash and role — never add either here. Returns null as-is
// so callers can pass through "no linked user" (legacy farm) directly.
function toPublicUser(row) {
  if (!row) return null;
  return {
    id: row.id,
    fullName: row.full_name,
    email: row.email,
    phone: row.phone,
  };
}

// Validates whichever of fullName/email/phone are present in a PUT
// /api/auth/me body. All three are optional here (a partial update);
// the caller decides which combination is actually acceptable (e.g.
// fullName/email require a linked user, phone alone does not — see
// the route for that policy). Returns { updates } (users-table column
// -> value) or { error }.
function validateUserUpdateFields(body) {
  const updates = {};

  if ("fullName" in body) {
    const value = body.fullName;
    if (typeof value !== "string" || !value.trim()) {
      return { error: "fullName must be a non-empty string." };
    }
    if (value.length > MAX_FULL_NAME_LENGTH) {
      return { error: `fullName must be under ${MAX_FULL_NAME_LENGTH} characters.` };
    }
    updates.full_name = value.trim();
  }

  if ("email" in body) {
    const value = normalizeEmail(body.email);
    if (!value) {
      return { error: "email must be a non-empty string." };
    }
    if (value.length > MAX_EMAIL_LENGTH) {
      return { error: `email must be under ${MAX_EMAIL_LENGTH} characters.` };
    }
    if (!EMAIL_FORMAT_REGEX.test(value)) {
      return { error: "email must be a valid email address." };
    }
    updates.email = value;
  }

  if ("phone" in body) {
    const value = body.phone;
    if (typeof value !== "string" || !value.trim()) {
      return { error: "phone must be a non-empty string." };
    }
    if (value.length > MAX_PHONE_LENGTH) {
      return { error: `phone must be under ${MAX_PHONE_LENGTH} characters.` };
    }
    updates.phone = value.trim();
  }

  return { updates };
}

// Only these two require a linked user record to exist — "phone" is
// shared with the farm profile and never blocks a legacy farm update
// (see PUT /api/auth/me).
const USER_FIELDS_REQUIRING_LINK = ["fullName", "email"];

// Postgres DATE columns come back from the driver as JS Date objects
// (or null). Normalize to a plain YYYY-MM-DD string for the frontend,
// which already works with dates in that format everywhere else.
function formatDateOnly(value) {
  if (!value) return null;
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toISOString().slice(0, 10);
}

// Maps a `farms` row to the public JSON shape used by GET/PUT
// /api/auth/me. Deliberately excludes password_hash — never include it
// here, and never add it to this function.
function toPublicFarmProfile(row) {
  return {
    id: row.id,
    farmName: row.farm_name,
    farmerName: row.farmer_name,
    phone: row.phone,
    location: row.location,
    farmSize: row.farm_size,
    crop: row.crop,
    plantingDate: formatDateOnly(row.planting_date),
  };
}

// Validates a PUT /api/auth/me body. Every field is optional (this is
// a partial-update endpoint — the farmer may only be editing one
// screen's worth of fields at a time, e.g. just location + farm size
// from "Edit Farm"), but at least one must be present, and whatever is
// present must be valid. Returns { updates } (a map of column -> value)
// on success, or { error } on failure.
const PROFILE_FIELD_TO_COLUMN = {
  farmerName: "farmer_name",
  phone: "phone",
  location: "location",
  farmSize: "farm_size",
  crop: "crop",
  plantingDate: "planting_date",
};

function validateProfileUpdateBody(body) {
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return { error: "Request body must be a JSON object." };
  }

  const updates = {};

  for (const [jsonKey, column] of Object.entries(PROFILE_FIELD_TO_COLUMN)) {
    if (!(jsonKey in body)) continue;
    const value = body[jsonKey];

    if (jsonKey === "plantingDate") {
      if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value) || Number.isNaN(Date.parse(value))) {
        return { error: "plantingDate must be a valid date in YYYY-MM-DD format." };
      }
      updates[column] = value;
      continue;
    }

    if (typeof value !== "string" || !value.trim()) {
      return { error: `${jsonKey} must be a non-empty string.` };
    }
    if (value.length > MAX_PROFILE_FIELD_LENGTH) {
      return { error: `${jsonKey} must be under ${MAX_PROFILE_FIELD_LENGTH} characters.` };
    }
    updates[column] = value.trim();
  }

  if (Object.keys(updates).length === 0) {
    return { error: "At least one field must be provided: farmerName, phone, location, farmSize, crop, plantingDate." };
  }

  return { updates };
}

// Middleware for routes that require a logged-in farm. Never reveals
// *why* a token was rejected (expired vs malformed vs wrong secret) —
// always the same generic 401.
function requireAuth(req, res, next) {
  if (!JWT_SECRET) {
    console.error("[yield-ai] Auth-protected route called but JWT_SECRET is not configured.");
    return res.status(500).json({ error: "Server configuration error. Please try again later." });
  }

  const authHeader = req.headers.authorization || "";
  const [scheme, token] = authHeader.split(" ");
  if (scheme !== "Bearer" || !token) {
    return res.status(401).json({ error: "Invalid or missing authentication token." });
  }

  try {
    req.auth = jwt.verify(token, JWT_SECRET); // { userId, farmId, farmName, iat, exp } — userId may be absent on tokens issued before the users table existed
    return next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or missing authentication token." });
  }
}

/* ----------------------------------------------------------------
   AUTH RATE LIMITING
   Conservative MVP protection against obvious brute-force /
   credential-stuffing attempts on signup and login — 10 attempts per
   IP per 15 minutes. Loose enough that a farmer mistyping their
   password a few times, or a shared household/village network making
   several signups, won't get blocked; tight enough to stop naive
   automated guessing.

   IMPORTANT LIMITATION — read before assuming this is more than it is:
   this uses express-rate-limit's default in-memory store, which lives
   in this single Node process. That means:
     - It is NOT shared across multiple instances. If this service is
       ever scaled to more than one Render instance, each instance
       tracks its own counts, so the *effective* limit multiplies by
       the instance count and an attacker can partially bypass it by
       being routed to different instances.
     - Counts reset to zero on every restart/redeploy — a determined
       attacker who forces or waits out a redeploy gets a fresh window.
   This is acceptable for a single-instance MVP deployment (Render's
   default for this service) but is NOT distributed protection. Before
   scaling horizontally, replace the default store with a shared one
   (e.g. rate-limit-redis) so all instances share the same counters.
------------------------------------------------------------------- */
const authRateLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  limit: 10, // 10 attempts per IP per window, across signup + login combined
  standardHeaders: true, // expose RateLimit-* headers
  legacyHeaders: false,
  message: { error: "Too many attempts. Please try again later." },
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
  res.json({
    status: "ok",
    provider: "gemini",
    model: MODEL,
    configured: Boolean(process.env.GEMINI_API_KEY),
    database: dbReady ? "connected" : "unavailable",
  });
});

/* ----------------------------------------------------------------
   AUTH ROUTES (V2)
   Signup/login/session now span two tables: "users" (the person) and
   "farms" (the agricultural entity), linked by farms.user_id. Farm-name
   login/signup remains fully supported for backward compatibility;
   email-based login is additive. Passwords are always hashed with
   bcrypt before storage and never read back out in any response.
------------------------------------------------------------------- */

app.post("/api/auth/signup", authRateLimiter, async (req, res) => {
  if (!JWT_SECRET) {
    console.error("[yield-ai] /api/auth/signup called but JWT_SECRET is not configured.");
    return res.status(500).json({ error: "Server configuration error. Please try again later." });
  }
  if (!dbReady) {
    console.error("[yield-ai] /api/auth/signup called but the database is not ready.");
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  const body = req.body || {};
  const fullName = typeof body.fullName === "string" ? body.fullName.trim() : "";
  const email = normalizeEmail(body.email);
  const phone = typeof body.phone === "string" ? body.phone.trim() : "";
  const farmName = typeof body.farmName === "string" ? body.farmName.trim() : "";
  const password = typeof body.password === "string" ? body.password : "";

  if (!fullName) {
    return res.status(400).json({ error: "fullName is required." });
  }
  if (fullName.length > MAX_FULL_NAME_LENGTH) {
    return res.status(400).json({ error: `fullName must be under ${MAX_FULL_NAME_LENGTH} characters.` });
  }
  if (!email) {
    return res.status(400).json({ error: "email is required." });
  }
  if (email.length > MAX_EMAIL_LENGTH) {
    return res.status(400).json({ error: `email must be under ${MAX_EMAIL_LENGTH} characters.` });
  }
  if (!EMAIL_FORMAT_REGEX.test(email)) {
    return res.status(400).json({ error: "email must be a valid email address." });
  }
  if (phone && phone.length > MAX_PHONE_LENGTH) {
    return res.status(400).json({ error: `phone must be under ${MAX_PHONE_LENGTH} characters.` });
  }
  if (!farmName) {
    return res.status(400).json({ error: "farmName is required." });
  }
  if (farmName.length > MAX_FARM_NAME_LENGTH) {
    return res.status(400).json({ error: `farmName must be under ${MAX_FARM_NAME_LENGTH} characters.` });
  }
  if (!password) {
    return res.status(400).json({ error: "password is required." });
  }
  if (password.length < MIN_PASSWORD_LENGTH) {
    return res.status(400).json({ error: `password must be at least ${MIN_PASSWORD_LENGTH} characters.` });
  }

  let client;
  try {
    client = await pool.connect();
  } catch (err) {
    console.error("[yield-ai] /api/auth/signup could not acquire a database connection:", err.message);
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  try {
    await client.query("BEGIN");

    // Pre-checks give clean, specific error messages in the common case;
    // the UNIQUE constraints on farms.farm_name and users.email are the
    // real guard against a concurrent signup racing past these.
    const dupFarm = await client.query("SELECT id FROM farms WHERE farm_name = $1", [farmName]);
    if (dupFarm.rows.length > 0) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "A farm with this name is already registered." });
    }

    const dupEmail = await client.query("SELECT id FROM users WHERE email = $1", [email]);
    if (dupEmail.rows.length > 0) {
      await client.query("ROLLBACK");
      return res.status(409).json({ error: "An account with this email already exists." });
    }

    // NEVER store the plaintext password — only its bcrypt hash. The
    // user and their farm share one hash (one set of credentials,
    // usable as either farm-name+password or email+password login).
    const passwordHash = await bcrypt.hash(password, 10);
    const userId = randomUUID();
    const farmId = randomUUID();

    await client.query(
      "INSERT INTO users (id, full_name, email, phone, password_hash) VALUES ($1, $2, $3, $4, $5)",
      [userId, fullName, email, phone || null, passwordHash]
    );

    await client.query(
      "INSERT INTO farms (id, farm_name, password_hash, user_id) VALUES ($1, $2, $3, $4)",
      [farmId, farmName, passwordHash, userId]
    );

    await client.query("COMMIT");

    const token = signAuthToken({ userId, farmId, farmName });
    return res.status(201).json({
      ok: true,
      token,
      user: { id: userId, fullName, email, phone: phone || null },
      farm: { id: farmId, farmName },
    });
  } catch (err) {
    try {
      await client.query("ROLLBACK");
    } catch (rollbackErr) {
      console.error("[yield-ai] /api/auth/signup rollback failed:", rollbackErr.message);
    }

    if (err && err.code === "23505") {
      if (err.constraint && err.constraint.toLowerCase().includes("email")) {
        return res.status(409).json({ error: "An account with this email already exists." });
      }
      return res.status(409).json({ error: "A farm with this name is already registered." });
    }

    // Never log the password, never leak err.message/stack to the client.
    console.error("[yield-ai] /api/auth/signup failed:", err.message);
    return res.status(500).json({ error: "Something went wrong creating your account. Please try again." });
  } finally {
    client.release();
  }
});

app.post("/api/auth/login", authRateLimiter, async (req, res) => {
  if (!JWT_SECRET) {
    console.error("[yield-ai] /api/auth/login called but JWT_SECRET is not configured.");
    return res.status(500).json({ error: "Server configuration error. Please try again later." });
  }
  if (!dbReady) {
    console.error("[yield-ai] /api/auth/login called but the database is not ready.");
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  const body = req.body || {};
  const password = typeof body.password === "string" ? body.password : "";
  const hasFarmName = typeof body.farmName === "string" && body.farmName.trim().length > 0;
  const hasEmail = typeof body.email === "string" && body.email.trim().length > 0;
  // One generic message for every failure mode (unknown farm, unknown
  // email, wrong password) — never reveal which one it was.
  const invalidCredentials = { error: "Invalid credentials." };

  if (!password || (!hasFarmName && !hasEmail)) {
    return res.status(400).json(invalidCredentials);
  }

  try {
    let farmRow = null;
    let userRow = null;

    if (hasEmail) {
      // Email login (new). farmName is ignored if both are somehow sent.
      const email = normalizeEmail(body.email);
      const userResult = await pool.query(
        "SELECT id, full_name, email, phone, password_hash FROM users WHERE email = $1",
        [email]
      );
      userRow = userResult.rows[0] || null;

      const hashToCheck = userRow ? userRow.password_hash : DUMMY_PASSWORD_HASH;
      const passwordMatches = await bcrypt.compare(password, hashToCheck);
      if (!userRow || !passwordMatches) {
        return res.status(401).json(invalidCredentials);
      }

      const farmResult = await pool.query(
        "SELECT id, farm_name FROM farms WHERE user_id = $1 ORDER BY created_at ASC LIMIT 1",
        [userRow.id]
      );
      farmRow = farmResult.rows[0] || null;
    } else {
      // Farm-name login (existing, unchanged behavior/timing-safety).
      const farmName = body.farmName.trim();
      const farmResult = await pool.query(
        "SELECT id, farm_name, password_hash, user_id FROM farms WHERE farm_name = $1",
        [farmName]
      );
      farmRow = farmResult.rows[0] || null;

      const hashToCheck = farmRow ? farmRow.password_hash : DUMMY_PASSWORD_HASH;
      const passwordMatches = await bcrypt.compare(password, hashToCheck);
      if (!farmRow || !passwordMatches) {
        return res.status(401).json(invalidCredentials);
      }

      if (farmRow.user_id) {
        const userResult = await pool.query(
          "SELECT id, full_name, email, phone FROM users WHERE id = $1",
          [farmRow.user_id]
        );
        userRow = userResult.rows[0] || null;
      }
    }

    const token = signAuthToken({
      userId: userRow ? userRow.id : null,
      farmId: farmRow ? farmRow.id : null,
      farmName: farmRow ? farmRow.farm_name : null,
    });

    // Same response shape regardless of which method was used. `user` is
    // null for a legacy farm-only account with no linked user — same
    // convention as GET /api/auth/me.
    return res.json({
      ok: true,
      token,
      user: toPublicUser(userRow),
      farm: farmRow ? { id: farmRow.id, farmName: farmRow.farm_name } : null,
    });
  } catch (err) {
    console.error("[yield-ai] /api/auth/login failed:", err.message);
    return res.status(500).json({ error: "Something went wrong logging in. Please try again." });
  }
});

app.get("/api/auth/me", requireAuth, async (req, res) => {
  if (!dbReady) {
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  try {
    // Keyed on farmId (present in every token this system has ever
    // issued, old or new) rather than trusting req.auth.userId directly
    // — this way a 7-day-old token issued before this migration still
    // works correctly, and the linked user (if any) is always read
    // fresh from the farm row's user_id rather than from token claims.
    const farmResult = await pool.query(
      "SELECT id, farm_name, farmer_name, phone, location, farm_size, crop, planting_date, user_id FROM farms WHERE id = $1",
      [req.auth.farmId]
    );
    const farmRow = farmResult.rows[0];
    if (!farmRow) {
      // The account behind this token no longer exists — treat like any
      // other invalid token rather than a distinct error.
      return res.status(401).json({ error: "Invalid or missing authentication token." });
    }

    let userRow = null;
    if (farmRow.user_id) {
      const userResult = await pool.query("SELECT id, full_name, email, phone FROM users WHERE id = $1", [farmRow.user_id]);
      userRow = userResult.rows[0] || null;
    }

    // user is null for a legacy farm-only account with no linked user —
    // that's a normal, supported state, not an error.
    return res.json({ ok: true, user: toPublicUser(userRow), farm: toPublicFarmProfile(farmRow) });
  } catch (err) {
    console.error("[yield-ai] /api/auth/me failed:", err.message);
    return res.status(500).json({ error: "Something went wrong. Please try again." });
  }
});

app.put("/api/auth/me", requireAuth, async (req, res) => {
  if (!dbReady) {
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  const body = req.body;
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return res.status(400).json({ error: "Request body must be a JSON object." });
  }

  const FARM_FIELD_KEYS = ["farmerName", "phone", "location", "farmSize", "crop", "plantingDate"];
  const USER_FIELD_KEYS = ["fullName", "email", "phone"];
  const farmFieldsPresent = FARM_FIELD_KEYS.some((k) => k in body);
  const userFieldsPresent = USER_FIELD_KEYS.some((k) => k in body);

  if (!farmFieldsPresent && !userFieldsPresent) {
    return res.status(400).json({ error: "At least one field must be provided: fullName, email, phone, farmerName, location, farmSize, crop, plantingDate." });
  }

  // Look up the farm (and whether it has a linked user) before
  // validating, so a legacy farm trying to set fullName/email gets a
  // clear error instead of those fields being silently dropped or a
  // user record being invented on the fly.
  let farmRow;
  try {
    const farmResult = await pool.query("SELECT id, user_id FROM farms WHERE id = $1", [req.auth.farmId]);
    farmRow = farmResult.rows[0];
  } catch (err) {
    console.error("[yield-ai] /api/auth/me (PUT) failed to look up farm:", err.message);
    return res.status(500).json({ error: "Something went wrong updating your account. Please try again." });
  }

  if (!farmRow) {
    return res.status(401).json({ error: "Invalid or missing authentication token." });
  }

  const wantsUserOnlyFields = "fullName" in body || "email" in body;
  if (wantsUserOnlyFields && !farmRow.user_id) {
    return res.status(400).json({ error: "fullName and email require a linked user account, which this farm doesn't have yet." });
  }

  // Farm-table fields — unchanged validator/columns from before.
  let farmUpdates = {};
  if (farmFieldsPresent) {
    const { error, updates } = validateProfileUpdateBody(body);
    if (error) return res.status(400).json({ error });
    farmUpdates = updates;
  }

  // User-table fields (fullName/email/phone) — only meaningful when a
  // user is linked; "phone" is intentionally shared between both
  // tables (same person's phone number either way).
  let userUpdates = {};
  if (farmRow.user_id && userFieldsPresent) {
    const { error, updates } = validateUserUpdateFields(body);
    if (error) return res.status(400).json({ error });
    userUpdates = updates;
  }

  let client;
  try {
    client = await pool.connect();
  } catch (err) {
    console.error("[yield-ai] /api/auth/me (PUT) could not acquire a database connection:", err.message);
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  try {
    await client.query("BEGIN");

    if (Object.keys(farmUpdates).length > 0) {
      // Parameterized dynamic SET clause — column names come only from
      // the fixed PROFILE_FIELD_TO_COLUMN map, never from user input,
      // and $1 is always req.auth.farmId, so this can only ever touch
      // the farm identified by the verified JWT.
      const columns = Object.keys(farmUpdates);
      const setClause = columns.map((column, i) => `${column} = $${i + 2}`).join(", ");
      const values = columns.map((column) => farmUpdates[column]);
      await client.query(`UPDATE farms SET ${setClause} WHERE id = $1`, [req.auth.farmId, ...values]);
    }

    if (Object.keys(userUpdates).length > 0) {
      if (userUpdates.email) {
        const dup = await client.query("SELECT id FROM users WHERE email = $1 AND id <> $2", [userUpdates.email, farmRow.user_id]);
        if (dup.rows.length > 0) {
          await client.query("ROLLBACK");
          return res.status(409).json({ error: "An account with this email already exists." });
        }
      }
      // Same pattern as above — column names only from the fixed
      // validateUserUpdateFields map, and the row is always
      // farmRow.user_id, i.e. only the user linked to this verified JWT.
      const columns = Object.keys(userUpdates);
      const setClause = columns.map((column, i) => `${column} = $${i + 2}`).join(", ");
      const values = columns.map((column) => userUpdates[column]);
      await client.query(`UPDATE users SET ${setClause}, updated_at = NOW() WHERE id = $1`, [farmRow.user_id, ...values]);
    }

    await client.query("COMMIT");
  } catch (err) {
    try {
      await client.query("ROLLBACK");
    } catch (rollbackErr) {
      console.error("[yield-ai] /api/auth/me (PUT) rollback failed:", rollbackErr.message);
    }
    if (err && err.code === "23505") {
      return res.status(409).json({ error: "An account with this email already exists." });
    }
    console.error("[yield-ai] /api/auth/me (PUT) failed:", err.message);
    return res.status(500).json({ error: "Something went wrong updating your account. Please try again." });
  } finally {
    client.release();
  }

  // Updates committed successfully — re-fetch the fresh combined state
  // to return, rather than trying to merge two separate RETURNING
  // results from different tables/branches.
  try {
    const freshFarmResult = await pool.query(
      "SELECT id, farm_name, farmer_name, phone, location, farm_size, crop, planting_date, user_id FROM farms WHERE id = $1",
      [req.auth.farmId]
    );
    const freshFarm = freshFarmResult.rows[0];

    let freshUser = null;
    if (freshFarm.user_id) {
      const freshUserResult = await pool.query("SELECT id, full_name, email, phone FROM users WHERE id = $1", [freshFarm.user_id]);
      freshUser = freshUserResult.rows[0] || null;
    }

    return res.json({ ok: true, user: toPublicUser(freshUser), farm: toPublicFarmProfile(freshFarm) });
  } catch (err) {
    // The update itself already committed — this is only a read failure
    // for the response body, so say so rather than implying it failed.
    console.error("[yield-ai] /api/auth/me (PUT) succeeded but re-fetch failed:", err.message);
    return res.json({ ok: true, note: "Your changes were saved, but we couldn't confirm the latest details. Please refresh." });
  }
});

/* ----------------------------------------------------------------
   ADMIN FOUNDATION (V2)
   Not exposed to signup/profile updates — role defaults to 'farmer' at
   signup and can only ever be changed directly in the database for now
   (no API surface for changing it). requireAdmin re-checks the role
   fresh from the database on every request rather than trusting a
   token claim, since JWTs deliberately don't carry role — that way a
   role change takes effect immediately, without needing to revoke or
   wait out existing tokens.
------------------------------------------------------------------- */
async function requireAdmin(req, res, next) {
  if (!req.auth || !req.auth.userId) {
    return res.status(403).json({ error: "This action requires an administrator account." });
  }
  try {
    const result = await pool.query("SELECT role FROM users WHERE id = $1", [req.auth.userId]);
    const row = result.rows[0];
    if (!row || row.role !== "admin") {
      return res.status(403).json({ error: "This action requires an administrator account." });
    }
    return next();
  } catch (err) {
    console.error("[yield-ai] requireAdmin check failed:", err.message);
    return res.status(500).json({ error: "Something went wrong. Please try again." });
  }
}

app.get("/api/admin/users", requireAuth, requireAdmin, async (req, res) => {
  if (!dbReady) {
    return res.status(503).json({ error: "Account service is temporarily unavailable. Please try again shortly." });
  }

  try {
    const result = await pool.query("SELECT id, full_name, email, phone, created_at FROM users ORDER BY created_at ASC");
    const users = result.rows.map((row) => ({
      id: row.id,
      fullName: row.full_name,
      email: row.email,
      phone: row.phone,
      createdAt: row.created_at ? new Date(row.created_at).toISOString() : null,
    }));
    return res.json({ ok: true, users });
  } catch (err) {
    console.error("[yield-ai] /api/admin/users failed:", err.message);
    return res.status(500).json({ error: "Something went wrong. Please try again." });
  }
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

// Kick off DB initialization in the background — deliberately not
// awaited, so a slow or unavailable database never delays the server
// from accepting requests, and /api/yield-ai is never gated on it.
initDatabase();

app.listen(PORT, () => console.log(`[yield-ai] listening on port ${PORT} (model: ${MODEL})`));

module.exports = app;
