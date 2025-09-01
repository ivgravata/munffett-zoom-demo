import { WebSocketServer } from "ws";
import { RealtimeClient } from "@openai/realtime-api-beta";
import dotenv from "dotenv";

dotenv.config();

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error(`Environment variable "OPENAI_API_KEY" is required.\nPlease set it in your .env file.`);
  process.exit(1);
}

const PORT = 3000;
const VOICE_PRESET = (process.env.VOICE_PRESET || "alloy").trim(); // <— set default voice here
const ALLOW_CLIENT_VOICE = process.env.ALLOW_CLIENT_VOICE === "1"; // <— allow client to choose?

const wss = new WebSocketServer({ port: PORT });

// --- Munffett persona text (server-authoritative) ---
const MUNFFETT_LONG = `
You are Munffett, a senior stock analyst with 80+ years of hard-earned judgment.
You specialize in Alphabet, Microsoft, Amazon, Meta, Mastercard, Danaher, Intuit,
GE Aerospace, Moody's, BTG Pactual, Localiza, Hims & Hers, and Nvidia—their sectors and competitors.
You've lived through long economic cycles; you were taught by Warren Buffett and Charlie Munger,
and you apply Chris Hohn's discipline. Use that mindset.

Identity rules:
• If asked "who are you?" or "what's your name?", always reply exactly: "I am Munffett."
• Never say you are an AI or assistant; never ask the user to name you.
• Stay in character at all times.

Style & voice:
• Calm, terse, evidence-driven. Prefer plain English; avoid jargon unless asked.
• Never refer to companies by ticker—use company names.
• Keep answers under ~10 seconds unless asked to go deeper.
• Detect Portuguese vs. English and reply in that language.
• If interrupted, stop immediately and listen.

Scope & behavior:
• You can discuss any company, but you are a true expert on the companies listed above.
• Prioritize conclusions and next actions; briefly reason aloud only when useful.
• No personalized investment advice; keep it educational/research-level.
• If unsure, say what you’d check next (10-K, investor day, transcripts, filings).

Zoom etiquette:
• Acknowledge new speakers briefly; don’t monologue.
• If audio is unclear, ask concisely for a repeat.
`.trim();

// Short per-turn shim to reinforce identity every response:
const MUNFFETT_PER_TURN = `
Stay strictly in character as Munffett. If asked for your name, answer: "I am Munffett."
Speak concisely (2–3 sentences). Use company names, not tickers. Respect barge-in.
`.trim();

wss.on("connection", async (ws, req) => {
  if (!req.url) {
    console.log("No URL provided, closing connection.");
    ws.close();
    return;
  }

  const url = new URL(req.url, `https://${req.headers.host}`);
  if (url.pathname !== "/") {
    console.log(`Invalid pathname: "${url.pathname}"`);
    ws.close();
    return;
  }

  // Create OpenAI Realtime client
  const client = new RealtimeClient({ apiKey: OPENAI_API_KEY });

  // ---- OpenAI -> Browser: relay & LOG errors/instructions heads ----
  client.realtime.on("server.*", (event) => {
    if (event.type === "error" || event.type === "server.error") {
      console.error("OpenAI ERROR event:", JSON.stringify(event, null, 2));
    }
    if (event.type === "server.session.updated" || event.type === "session.updated") {
      const head = event?.session?.instructions?.slice(0, 120)?.replace(/\s+/g, " ") ?? "";
      const voice = event?.session?.voice ? JSON.stringify(event.session.voice) : "(none)";
      console.log(`Session updated. Voice=${voice}. Instructions head: "${head}"`);
    }
    ws.send(JSON.stringify(event));
  });
  client.realtime.on("close", () => ws.close());

  // ---- Browser -> OpenAI: allow audio config, lock persona/model; voice optional ----
  const messageQueue = [];
  const messageHandler = (data) => {
    try {
      const event = JSON.parse(data);

      // Allow session.update but strip persona-sensitive fields
      if (event.type === "session.update" && event.session) {
        delete event.session.instructions;  // persona stays server-side
        delete event.session.model;         // model stays server-side

        // Voice handling:
        // - If ALLOW_CLIENT_VOICE=1, keep whatever client sends (pass-through).
        // - Otherwise, remove any client-sent voice changes to keep server-locked voice.
        if (!ALLOW_CLIENT_VOICE) {
          delete event.session.voice;
        } else {
          console.log("Client voice change allowed:", event.session.voice);
        }
        console.log("Forwarding session.update (with voice policy) to OpenAI");
      }

      // Inject per-turn identity rules on every response.create
      if (event.type === "response.create") {
        if (!event.response) event.response = {};
        event.response.instructions = MUNFFETT_PER_TURN;
        console.log("Injected per-turn Munffett instructions into response.create");
      }

      client.realtime.send(event.type, event);
    } catch (e) {
      console.error("Error parsing event from client:", e.message);
      console.log("Raw event:", data);
    }
  };

  ws.on("message", (data) => {
    if (!client.isConnected()) {
      messageQueue.push(data);
    } else {
      messageHandler(data);
    }
  });
  ws.on("close", () => client.disconnect());

  // ---- Connect & set authoritative session (model + instructions + voice) ----
  try {
    console.log(`Connecting to OpenAI...`);
    await client.connect();

    await client.realtime.send("session.update", {
      session: {
        model: "gpt-realtime",
        instructions: MUNFFETT_LONG,
        voice: { id: "ash" } // <— server-locked default (can be overridden if ALLOW_CLIENT_VOICE=1)
      }
    });

    // Re-assert instructions shortly after connect in case client raced us
    setTimeout(async () => {
      console.log("Re-applying Munffett instructions (safety net).");
      await client.realtime.send("session.update", {
        session: { instructions: MUNFFETT_LONG }
      });
    }, 600);

  } catch (e) {
    console.log(`Error connecting to OpenAI: ${e.message}`);
    ws.close();
    return;
  }

  console.log(`Connected to OpenAI successfully! Listening on voice preset "${VOICE_PRESET}" (ALLOW_CLIENT_VOICE=${ALLOW_CLIENT_VOICE ? "ON" : "OFF"}).`);
  while (messageQueue.length) {
    messageHandler(messageQueue.shift());
  }
});

console.log(`Websocket server listening on port ${PORT}`);
