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
        self.base_url = "https://us-west-2.recall.ai/api/v1"
        
    async def create_bot(self, meeting_url: str, bot_name: str = "AI Assistant", 
                        persona_key: str = "assistant") -> Dict[str, Any]:
        """Create a bot in Recall.ai."""
        # Get the WebSocket URL for this persona
        ws_url = f"{PUBLIC_URL.replace('https://', 'wss://').replace('http://', 'ws://')}/ws?persona={persona_key}"
        
        payload = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
            "output_media": {
                "camera": {
                    "kind": "webpage",
                    "config": {
                        "url": f"{PUBLIC_URL}/agent?wss={ws_url}"
                    }
                }
            }
        }
        
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
                        "type": "server_vad"
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
        
        # Relay messages between client and OpenAI
        async def relay_to_openai():
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        event = json.loads(msg.data)
                        logger.info(f'Relaying "{event.get("type")}" to OpenAI')
                        await openai_ws.send(msg.data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON from client: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    break
        
        async def relay_from_openai():
            while True:
                try:
                    message = await openai_ws.recv()
                    event = json.loads(message)
                    logger.info(f'Relaying "{event.get("type")}" from OpenAI')
                    await ws.send_str(message)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.error(f"Error relaying from OpenAI: {e}")
                    break
        
        # Run both relay tasks concurrently
        await asyncio.gather(relay_to_openai(), relay_from_openai())
        
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        if openai_ws and not openai_ws.closed:
            await openai_ws.close()
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
    """Serve the agent HTML page."""
    # Get WebSocket URL from query parameter
    wss_url = request.query.get('wss', '')
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Agent</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: white;
        }}
        .container {{
            text-align: center;
            padding: 20px;
        }}
        h1 {{
            font-size: 48px;
            margin-bottom: 10px;
        }}
        .status {{
            font-size: 24px;
            margin: 20px 0;
        }}
        .listening {{
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        #error {{
            background: rgba(255, 0, 0, 0.2);
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 AI Voice Agent</h1>
        <div class="status">
            <div id="status-text">Connecting to OpenAI...</div>
        </div>
        <div id="error"></div>
    </div>

    <script>
        const wssUrl = '{wss_url}';
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let audioQueue = [];
        let isPlaying = false;

        function updateStatus(text, isListening = false) {{
            const statusDiv = document.getElementById('status-text');
            statusDiv.textContent = text;
            statusDiv.className = isListening ? 'listening' : '';
        }}

        function showError(message) {{
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = 'Error: ' + message;
            errorDiv.style.display = 'block';
        }}

        async function initializeAudio() {{
            try {{
                // Get microphone access
                mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                audioContext = new (window.AudioContext || window.webkitAudioContext)({{ sampleRate: 24000 }});
                
                // Set up audio processing
                const source = audioContext.createMediaStreamSource(mediaStream);
                const processor = audioContext.createScriptProcessor(4096, 1, 1);
                
                processor.onaudioprocess = (e) => {{
                    if (ws && ws.readyState === WebSocket.OPEN) {{
                        const inputData = e.inputBuffer.getChannelData(0);
                        // Convert to 16-bit PCM
                        const pcm16 = new Int16Array(inputData.length);
                        for (let i = 0; i < inputData.length; i++) {{
                            pcm16[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                        }}
                        
                        // Send audio to OpenAI
                        ws.send(JSON.stringify({{
                            type: 'input_audio_buffer.append',
                            audio: btoa(String.fromCharCode(...new Uint8Array(pcm16.buffer)))
                        }}));
                    }}
                }};
                
                source.connect(processor);
                processor.connect(audioContext.destination);
                
                updateStatus('✅ Audio initialized', true);
                return true;
            }} catch (error) {{
                showError('Failed to access microphone: ' + error.message);
                return false;
            }}
        }}

        async function connectWebSocket() {{
            if (!wssUrl) {{
                showError('No WebSocket URL provided');
                return;
            }}

            try {{
                updateStatus('Connecting to AI...');
                ws = new WebSocket(wssUrl);
                
                ws.onopen = async () => {{
                    console.log('Connected to WebSocket');
                    updateStatus('Connected! Initializing audio...');
                    await initializeAudio();
                    updateStatus('🎤 Listening... Start talking!', true);
                }};
                
                ws.onmessage = (event) => {{
                    try {{
                        const data = JSON.parse(event.data);
                        console.log('Received:', data.type);
                        
                        if (data.type === 'response.audio.delta' && data.delta) {{
                            // Handle audio response from OpenAI
                            playAudioDelta(data.delta);
                        }} else if (data.type === 'response.audio_transcript.done') {{
                            console.log('AI said:', data.transcript);
                        }} else if (data.type === 'conversation.item.input_audio_transcription.completed') {{
                            console.log('You said:', data.transcript);
                        }}
                    }} catch (error) {{
                        console.error('Error processing message:', error);
                    }}
                }};
                
                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    showError('Connection error');
                }};
                
                ws.onclose = () => {{
                    console.log('WebSocket closed');
                    updateStatus('Disconnected. Refreshing...');
                    setTimeout(() => location.reload(), 3000);
                }};
            }} catch (error) {{
                showError('Failed to connect: ' + error.message);
            }}
        }}

        async function playAudioDelta(base64Audio) {{
            try {{
                // Decode base64 to PCM16
                const binaryString = atob(base64Audio);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {{
                    bytes[i] = binaryString.charCodeAt(i);
                }}
                
                // Convert to Float32 for Web Audio
                const pcm16 = new Int16Array(bytes.buffer);
                const float32 = new Float32Array(pcm16.length);
                for (let i = 0; i < pcm16.length; i++) {{
                    float32[i] = pcm16[i] / 32768;
                }}
                
                // Create and play audio buffer
                const audioBuffer = audioContext.createBuffer(1, float32.length, 24000);
                audioBuffer.getChannelData(0).set(float32);
                
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
            }} catch (error) {{
                console.error('Error playing audio:', error);
            }}
        }}

        // Start connection
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
    web.run_app(app, host='0.0.0.0', port=PORT)