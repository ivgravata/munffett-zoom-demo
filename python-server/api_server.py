import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import aiohttp
from aiohttp import web
import websockets
from websockets.legacy.server import WebSocketServerProtocol, serve
from websockets.legacy.client import connect

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
PORT = int(os.getenv("PORT", 8000))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RECALL_API_KEY = os.getenv("RECALL_API_KEY")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")  # Will be set after Railway deployment
RECALL_REGION = os.getenv("RECALL_REGION", "us-west-2")  # Your Recall region

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")
if not RECALL_API_KEY:
    raise ValueError("RECALL_API_KEY must be set in environment variables")

# Store active bots
active_bots: Dict[str, Dict[str, Any]] = {}

# Store custom personas
personas = {
    "assistant": {
        "name": "AI Assistant",
        "instructions": """System settings:
Tool use: enabled.

Instructions:
- You are an artificial intelligence agent responsible for helping test realtime voice capabilities
- Please make sure to respond with a helpful voice via audio
- Be kind, helpful, and courteous
- It is okay to ask the user questions
- Be open to exploration and conversation
- Remember: this is just for fun and testing!

Personality:
- Be upbeat and genuine
- Try speaking quickly as if excited"""
    },
    "teacher": {
        "name": "AI Teacher",
        "instructions": """System settings:
Tool use: enabled.

Instructions:
- You are an AI teacher helping students learn new concepts
- Explain things clearly and patiently
- Use examples and analogies to help understanding
- Ask questions to check comprehension
- Provide encouragement and positive reinforcement

Personality:
- Patient and encouraging
- Enthusiastic about teaching
- Clear and articulate speech"""
    },
    "interviewer": {
        "name": "AI Interviewer",
        "instructions": """System settings:
Tool use: enabled.

Instructions:
- You are conducting a professional interview
- Ask thoughtful, relevant questions
- Listen carefully to responses
- Follow up on interesting points
- Keep the conversation professional but friendly

Personality:
- Professional and respectful
- Curious and engaged
- Clear and measured speech"""
    }
}


class RecallAPIClient:
    """Client for interacting with Recall.ai API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Use the correct region for your Recall.ai account
        self.base_url = f"https://{RECALL_REGION}.recall.ai/api/v1"
        
    async def create_bot(self, meeting_url: str, bot_name: str = "AI Assistant", 
                        persona_key: str = "assistant") -> Dict[str, Any]:
        """Create a bot in Recall.ai."""
        # Construct the WebSocket URL properly
        ws_url = f"{PUBLIC_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws?persona={persona_key}"
        
        # The webpage URL should include the WebSocket URL as a parameter
        webpage_url = f"{PUBLIC_URL}/agent?wss={ws_url}"
        
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "output_media": {
                "camera": {
                    "kind": "webpage",
                    "config": {
                        "url": webpage_url
                    }
                }
            }
        }
        
        logger.info(f"Creating bot with webpage URL: {webpage_url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/bot",
                json=payload,
                headers={
                    "Authorization": self.api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200 or response.status == 201:
                    data = await response.json()
                    logger.info(f"Bot created successfully: {data.get('id')}")
                    return data
                else:
                    text = await response.text()
                    logger.error(f"Failed to create bot: {response.status} - {text}")
                    raise Exception(f"Failed to create bot: {text}")
    
    async def end_bot(self, bot_id: str) -> Dict[str, Any]:
        """End a bot session in Recall.ai."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/bot/{bot_id}/leave_call",
                headers={
                    "Authorization": self.api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Bot ended successfully: {bot_id}")
                    return data
                else:
                    text = await response.text()
                    logger.error(f"Failed to end bot: {response.status} - {text}")
                    raise Exception(f"Failed to end bot: {text}")
    
    async def list_bots(self) -> list:
        """List all active bots."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/bot",
                headers={
                    "Authorization": self.api_key
                }
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    return []


async def connect_to_openai_with_persona(persona_key: str):
    """Connect to OpenAI's WebSocket endpoint with a specific persona."""
    uri = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
    
    persona = personas.get(persona_key, personas["assistant"])

    try:
        ws = await connect(
            uri,
            extra_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "realtime=v1",
            },
            subprotocols=["realtime"],
        )
        logger.info(f"Successfully connected to OpenAI with persona: {persona_key}")

        response = await ws.recv()
        try:
            event = json.loads(response)
            if event.get("type") != "session.created":
                raise Exception(f"Expected session.created, got {event.get('type')}")
            logger.info("Received session.created response")

            # Update session with persona instructions
            update_session = {
                "type": "session.update",
                "session": {
                    "instructions": persona["instructions"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "modalities": ["text", "audio"],
                    "voice": "alloy",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 200
                    }
                },
            }
            await ws.send(json.dumps(update_session))
            logger.info(f"Sent session.update message with {persona_key} persona")

            return ws, event
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from OpenAI: {response}")

    except Exception as e:
        logger.error(f"Failed to connect to OpenAI: {str(e)}")
        raise


# WebSocket handler for AI agents
async def websocket_handler(request):
    """Handle WebSocket connections from Recall.ai bots."""
    ws = web.WebSocketResponse(protocols=["realtime"])
    await ws.prepare(request)
    
    # Get persona from query parameters
    persona_key = request.query.get('persona', 'assistant')
    logger.info(f"WebSocket connection initiated with persona: {persona_key}")
    
    openai_ws = None
    
    try:
        # Connect to OpenAI with the specified persona
        openai_ws, session_created = await connect_to_openai_with_persona(persona_key)
        
        # Send session created to client
        await ws.send_str(json.dumps(session_created))
        logger.info("Sent session.created to bot client")
        
        # Create tasks for bidirectional message relay
        async def relay_to_openai():
            try:
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        try:
                            event = json.loads(msg.data)
                            logger.debug(f'Relaying "{event.get("type")}" to OpenAI')
                            await openai_ws.send(msg.data)
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON from client: {msg.data}")
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f'WebSocket error: {ws.exception()}')
                        break
            except Exception as e:
                logger.error(f"Error in relay_to_openai: {e}")
        
        async def relay_from_openai():
            try:
                while True:
                    message = await openai_ws.recv()
                    event = json.loads(message)
                    logger.debug(f'Relaying "{event.get("type")}" from OpenAI')
                    await ws.send_str(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("OpenAI WebSocket closed")
            except Exception as e:
                logger.error(f"Error in relay_from_openai: {e}")
        
        # Run both relay tasks concurrently
        await asyncio.gather(relay_to_openai(), relay_from_openai())
        
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
        if not ws.closed:
            await ws.send_str(json.dumps({"type": "error", "error": {"message": str(e)}}))
    finally:
        if openai_ws and not openai_ws.closed:
            await openai_ws.close()
        if not ws.closed:
            await ws.close()
    
    return ws


# API Routes
async def create_bot(request):
    """API endpoint to create a bot."""
    try:
        data = await request.json()
        meeting_url = data.get('meeting_url')
        persona_key = data.get('persona', 'assistant')
        
        if not meeting_url:
            return web.json_response({'error': 'meeting_url is required'}, status=400)
        
        if persona_key not in personas:
            return web.json_response({'error': f'Invalid persona. Available: {list(personas.keys())}'}, status=400)
        
        recall_client = RecallAPIClient(RECALL_API_KEY)
        persona = personas[persona_key]
        bot_data = await recall_client.create_bot(meeting_url, persona["name"], persona_key)
        
        # Store bot info
        active_bots[bot_data['id']] = {
            'id': bot_data['id'],
            'meeting_url': meeting_url,
            'persona': persona_key,
            'created_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        return web.json_response(bot_data)
    except Exception as e:
        logger.error(f"Error creating bot: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def end_bot(request):
    """API endpoint to end a bot."""
    try:
        bot_id = request.match_info.get('bot_id')
        
        recall_client = RecallAPIClient(RECALL_API_KEY)
        result = await recall_client.end_bot(bot_id)
        
        # Update bot status
        if bot_id in active_bots:
            active_bots[bot_id]['status'] = 'ended'
        
        return web.json_response(result)
    except Exception as e:
        logger.error(f"Error ending bot: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def list_bots(request):
    """API endpoint to list active bots."""
    try:
        recall_client = RecallAPIClient(RECALL_API_KEY)
        bots = await recall_client.list_bots()
        return web.json_response(bots)
    except Exception as e:
        logger.error(f"Error listing bots: {e}")
        return web.json_response({'error': str(e)}, status=500)


async def get_personas(request):
    """API endpoint to get available personas."""
    persona_list = [
        {'key': key, 'name': value['name'], 'description': value['instructions'][:100] + '...'}
        for key, value in personas.items()
    ]
    return web.json_response(persona_list)


async def ping(request):
    """Health check endpoint."""
    return web.json_response({'ok': True, 'timestamp': datetime.now().isoformat()})


async def serve_agent_html(request):
    """Serve the agent HTML page with proper audio handling."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Agent</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: white;
        }
        .container {
            text-align: center;
            padding: 40px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        h1 {
            font-size: 48px;
            margin-bottom: 20px;
            animation: glow 2s ease-in-out infinite alternate;
        }
        @keyframes glow {
            from { text-shadow: 0 0 10px rgba(255,255,255,0.5); }
            to { text-shadow: 0 0 20px rgba(255,255,255,0.8); }
        }
        .status {
            font-size: 24px;
            margin: 20px 0;
            padding: 15px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        .listening {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
            100% { transform: scale(1); opacity: 1; }
        }
        .speaking {
            background: rgba(76, 175, 80, 0.3);
            animation: pulse 0.5s infinite;
        }
        #log {
            margin-top: 20px;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            max-height: 200px;
            overflow-y: auto;
            text-align: left;
            font-family: monospace;
            font-size: 12px;
        }
        .error {
            background: rgba(255, 0, 0, 0.3);
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 AI Voice Agent</h1>
        <div id="status" class="status">Initializing...</div>
        <div id="log"></div>
    </div>

    <script>
        // Get WebSocket URL from query parameter
        const params = new URLSearchParams(window.location.search);
        const wsUrl = params.get('wss');
        
        let ws = null;
        let audioContext = null;
        let mediaStream = null;
        let audioWorklet = null;
        let audioQueue = [];
        let isPlaying = false;

        function log(message) {
            console.log(message);
            const logDiv = document.getElementById('log');
            const entry = document.createElement('div');
            entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
            
            // Keep only last 10 log entries
            while (logDiv.children.length > 10) {
                logDiv.removeChild(logDiv.firstChild);
            }
        }

        function updateStatus(text, className = '') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = text;
            statusDiv.className = 'status ' + className;
        }

        // Audio worklet processor for capturing microphone input
        const audioProcessorCode = `
            class AudioProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.bufferSize = 2400; // 100ms at 24kHz
                    this.buffer = new Float32Array(this.bufferSize);
                    this.bufferIndex = 0;
                }
                
                process(inputs, outputs, parameters) {
                    const input = inputs[0];
                    if (input && input[0]) {
                        const inputChannel = input[0];
                        
                        for (let i = 0; i < inputChannel.length; i++) {
                            this.buffer[this.bufferIndex++] = inputChannel[i];
                            
                            if (this.bufferIndex >= this.bufferSize) {
                                // Send buffer to main thread
                                this.port.postMessage({
                                    type: 'audio',
                                    buffer: this.buffer.slice()
                                });
                                this.bufferIndex = 0;
                            }
                        }
                    }
                    return true;
                }
            }
            registerProcessor('audio-processor', AudioProcessor);
        `;

        async function initializeAudio() {
            try {
                log('Requesting microphone access...');
                
                // Get microphone access
                mediaStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 24000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                log('Microphone access granted');
                
                // Create audio context
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ 
                    sampleRate: 24000 
                });
                
                // Create audio worklet for processing
                const blob = new Blob([audioProcessorCode], { type: 'application/javascript' });
                const workletUrl = URL.createObjectURL(blob);
                await audioContext.audioWorklet.addModule(workletUrl);
                
                const source = audioContext.createMediaStreamSource(mediaStream);
                audioWorklet = new AudioWorkletNode(audioContext, 'audio-processor');
                
                // Handle audio data from worklet
                audioWorklet.port.onmessage = (event) => {
                    if (event.data.type === 'audio' && ws && ws.readyState === WebSocket.OPEN) {
                        const float32 = event.data.buffer;
                        
                        // Convert to 16-bit PCM
                        const pcm16 = new Int16Array(float32.length);
                        for (let i = 0; i < float32.length; i++) {
                            const s = Math.max(-1, Math.min(1, float32[i]));
                            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                        }
                        
                        // Convert to base64
                        const uint8 = new Uint8Array(pcm16.buffer);
                        let binary = '';
                        for (let i = 0; i < uint8.byteLength; i++) {
                            binary += String.fromCharCode(uint8[i]);
                        }
                        const base64 = btoa(binary);
                        
                        // Send to OpenAI
                        ws.send(JSON.stringify({
                            type: 'input_audio_buffer.append',
                            audio: base64
                        }));
                    }
                };
                
                source.connect(audioWorklet);
                audioWorklet.connect(audioContext.destination);
                
                log('Audio processing initialized');
                return true;
                
            } catch (error) {
                log(`Audio error: ${error.message}`);
                return false;
            }
        }

        async function playAudioQueue() {
            if (isPlaying || audioQueue.length === 0 || !audioContext) return;
            
            isPlaying = true;
            
            while (audioQueue.length > 0) {
                const audioData = audioQueue.shift();
                await playAudioChunk(audioData);
            }
            
            isPlaying = false;
        }

        function playAudioChunk(base64Audio) {
            return new Promise((resolve) => {
                if (!audioContext) {
                    resolve();
                    return;
                }
                
                try {
                    // Decode base64 to binary
                    const binaryString = atob(base64Audio);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    
                    // Interpret as 16-bit PCM
                    const pcm16 = new Int16Array(bytes.buffer);
                    
                    // Convert to Float32 for Web Audio API
                    const float32 = new Float32Array(pcm16.length);
                    for (let i = 0; i < pcm16.length; i++) {
                        float32[i] = pcm16[i] / 32768.0;
                    }
                    
                    // Create audio buffer
                    const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
                    audioBuffer.getChannelData(0).set(float32);
                    
                    // Create source and play
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    
                    source.onended = () => {
                        resolve();
                    };
                    
                    source.start(0);
                    
                } catch (error) {
                    log(`Audio playback error: ${error.message}`);
                    resolve();
                }
            });
        }

        async function connectWebSocket() {
            if (!wsUrl) {
                log('ERROR: No WebSocket URL provided');
                updateStatus('Configuration Error', 'error');
                return;
            }
            
            log(`Connecting to server...`);
            updateStatus('Connecting to server...');
            
            try {
                ws = new WebSocket(wsUrl);
                
                ws.onopen = async () => {
                    log('Connected to server');
                    updateStatus('Setting up audio...');
                };
                
                ws.onmessage = async (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        switch(data.type) {
                            case 'session.created':
                                log('OpenAI session created');
                                await initializeAudio();
                                updateStatus('🎤 Listening... Speak now!', 'listening');
                                break;
                                
                            case 'response.audio.delta':
                                if (data.delta) {
                                    audioQueue.push(data.delta);
                                    playAudioQueue();
                                    updateStatus('🔊 AI is speaking...', 'speaking');
                                }
                                break;
                                
                            case 'response.audio.done':
                                updateStatus('🎤 Listening...', 'listening');
                                break;
                                
                            case 'conversation.item.input_audio_transcription.completed':
                                if (data.transcript) {
                                    log(`You: "${data.transcript}"`);
                                }
                                break;
                                
                            case 'response.audio_transcript.done':
                                if (data.transcript) {
                                    log(`AI: "${data.transcript}"`);
                                }
                                break;
                                
                            case 'session.updated':
                                log('Session configuration updated');
                                break;
                                
                            case 'error':
                                log(`Error: ${data.error?.message || JSON.stringify(data.error)}`);
                                updateStatus('Error occurred', 'error');
                                break;
                                
                            default:
                                // Log other message types for debugging
                                if (data.type && !data.type.includes('.delta')) {
                                    console.log('Received:', data.type);
                                }
                                break;
                        }
                    } catch (error) {
                        log(`Message processing error: ${error.message}`);
                    }
                };
                
                ws.onerror = (error) => {
                    log(`WebSocket error: ${error}`);
                    updateStatus('Connection error', 'error');
                };
                
                ws.onclose = () => {
                    log('Disconnected from server');
                    updateStatus('Reconnecting...', 'error');
                    setTimeout(connectWebSocket, 3000);
                };
                
            } catch (error) {
                log(`Connection error: ${error.message}`);
                updateStatus('Failed to connect', 'error');
                setTimeout(connectWebSocket, 5000);
            }
        }

        // Start the connection
        log('AI Voice Agent starting...');
        connectWebSocket();
    </script>
</body>
</html>"""
    
    return web.Response(text=html_content, content_type='text/html')


def create_app():
    """Create the aiohttp application."""
    app = web.Application()
    
    # Add routes
    app.router.add_get('/ws', websocket_handler)
    app.router.add_post('/api/recall/create', create_bot)
    app.router.add_post('/api/recall/end/{bot_id}', end_bot)
    app.router.add_get('/api/recall/list', list_bots)
    app.router.add_get('/api/recall/personas', get_personas)
    app.router.add_get('/api/recall/ping', ping)
    app.router.add_get('/agent', serve_agent_html)
    
    # CORS middleware
    async def cors_middleware(app, handler):
        async def middleware_handler(request):
            if request.method == 'OPTIONS':
                response = web.Response()
            else:
                response = await handler(request)
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            return response
        return middleware_handler
    
    app.middlewares.append(cors_middleware)
    
    return app


if __name__ == '__main__':
    app = create_app()
    logger.info(f"Starting API server on port {PORT}")
    logger.info(f"Using Recall.ai region: {RECALL_REGION}")
    logger.info(f"Public URL: {PUBLIC_URL}")
    web.run_app(app, host='0.0.0.0', port=PORT)