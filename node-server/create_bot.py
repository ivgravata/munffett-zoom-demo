import requests

url = "https://us-west-2.recall.ai/api/v1/bot/"
headers = {
    "Authorization": "239f73d2d115c2482a5875baf4b08601522b0318",
    "Content-Type": "application/json"
}
data = {
    "meeting_url": "https://us02web.zoom.us/j/9950557478?pwd=SkhYU3l6MDBmUmZUYTBBNEtOcXVadz09",
    "bot_name": "Zoom Voice Agent",
    "output_media": {
        "camera": {
            "kind": "webpage",
            "config": {
                "url": "https://recallai-demo.netlify.app?wss=wss://4b8f4eef1f70.ngrok-free.app"
            }
        }
    },
    "variant": {"zoom": "web_4_core"}
}

resp = requests.post(url, headers=headers, json=data)
print(resp.json())
