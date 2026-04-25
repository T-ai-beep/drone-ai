import subprocess
import json
import time
import requests

def capture_frame():
    result = subprocess.run([
        'ffmpeg', '-f', 'avfoundation',
        '-framerate', '30',
        '-video_size', '640x480',
        '-i', '0',
        '-vframes', '1',
        '-update', '1',
        '/tmp/gesture_frame.jpg',
        '-y'
    ], capture_output=True)
    return result.returncode == 0

def analyze_gesture():
    import base64
    with open('/tmp/gesture_frame.jpg', 'rb') as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = requests.post('http://localhost:11434/api/chat', json={
        "model": "llava",
        "messages": [{
            "role": "user",
            "content": "Look at this hand. Reply with ONLY one word: 'point' if index finger is extended, 'scan' if all fingers open, 'orbit' if fist, or 'none'.",
            "images": [image_data]
        }],
        "stream": False
    })
    
    text = response.json()['message']['content'].strip().lower()
    
    for gesture in ['point', 'scan', 'orbit']:
        if gesture in text:
            return gesture
    return 'none'

while True:
    if capture_frame():
        gesture = analyze_gesture()
        print(json.dumps({"gesture": gesture}), flush=True)
    else:
        print(json.dumps({"gesture": "none"}), flush=True)
    time.sleep(0.5)