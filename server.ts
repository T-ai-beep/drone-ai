import { serve } from "bun";
import { execSync, spawn } from "child_process";
import * as fs from "fs";

const FRAME_PATH = "/Users/tanayshah/drone/snapshot.jpg";
const clients = new Set<WebSocket>();

// Capture frame from camera
function captureFrame(): string | null {
  try {
    execSync(`ffmpeg -f avfoundation -pixel_format uyvy422 -framerate 30 -video_size 640x480 -i "0" -vframes 1 -update 1 ${FRAME_PATH} -y 2>/dev/null`);
    const data = fs.readFileSync(FRAME_PATH);
    return data.toString("base64");
  } catch {
    return null;
  }
}

// Get telemetry from MAVLink via Python
function getTelemetry(): object {
  try {
    const result = execSync(`python3 -c "
from pymavlink import mavutil
import json, sys
try:
    m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    m.wait_heartbeat(timeout=2)
    alt_msg = m.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
    att_msg = m.recv_match(type='ATTITUDE', blocking=True, timeout=1)
    bat_msg = m.recv_match(type='BATTERY_STATUS', blocking=True, timeout=1)
    import math
    data = {
        'altitude': round(alt_msg.relative_alt / 1000.0, 1) if alt_msg else 0,
        'heading': round(math.degrees(att_msg.yaw) % 360, 1) if att_msg else 0,
        'pitch': round(math.degrees(att_msg.pitch), 1) if att_msg else 0,
        'roll': round(math.degrees(att_msg.roll), 1) if att_msg else 0,
        'battery': bat_msg.battery_remaining if bat_msg else 75,
        'lat': alt_msg.lat / 1e7 if alt_msg else -35.363,
        'lon': alt_msg.lon / 1e7 if alt_msg else 149.165,
    }
    print(json.dumps(data))
except Exception as e:
    print(json.dumps({'altitude': 0, 'heading': 0, 'battery': 75, 'lat': -35.363, 'lon': 149.165, 'error': str(e)}))
"`, { timeout: 5000 });
    return JSON.parse(result.toString());
  } catch {
    return { altitude: 0, heading: 0, battery: 75, lat: -35.363, lon: 149.165 };
  }
}

// Get AI agent analysis
function getAgentAnalysis(base64Image: string): Promise<object> {
  return fetch("http://localhost:11434/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "llava",
      messages: [{
        role: "user",
        content: `You are an autonomous drone AI. Analyze this scene briefly.
Respond ONLY with JSON:
{
  "observation": "one sentence",
  "threat": "none/low/medium/high",
  "action": "hover/scan/move/orbit/land",
  "direction": "left/right/forward/backward/none"
}`,
        images: [base64Image]
      }],
      stream: false
    })
  })
  .then(r => r.json())
  .then(data => {
    const text = data.message.content;
    const start = text.indexOf('{');
    const end = text.lastIndexOf('}') + 1;
    return JSON.parse(text.slice(start, end));
  })
  .catch(() => ({ observation: "Analyzing...", threat: "none", action: "hover", direction: "none" }));
}

// Broadcast to all connected clients
function broadcast(data: object) {
  const msg = JSON.stringify(data);
  clients.forEach(ws => {
    if (ws.readyState === 1) ws.send(msg);
  });
}

function getAdvisorRecommendation(): object {
  try {
    const result = execSync(`python3 -c "
import sys
sys.path.insert(0, '/Users/tanayshah/drone')
from advisor import get_recommendation
import json

consensus_map = {
    'scenario': 'wildfire — Allen TX Sector 4',
    'zones': [{'id': 'ZONE-7', 'status': 'RED', 'coords': (33.1584, -96.6735), 'flagged_by': ['FARSITE', 'Rothermel', 'WindNinja']}]
}
thermal = [{'id': 'THERM-001', 'label': 'survivor_candidate', 'confidence': 0.91, 'bearing_deg': 47}]
fleet = {'DRONE-1': {'type': 'quadcopter', 'battery_pct': 75, 'status': 'active'}}

rec, reasons, conf, ts = get_recommendation(consensus_map, thermal, fleet)
print(json.dumps({'recommendation': rec, 'reasons': reasons, 'confidence': conf, 'timestamp': ts}))
"`, { timeout: 60000 });
    return JSON.parse(result.toString());
  } catch (e) {
    return { recommendation: "Advisor unavailable", reasons: [], confidence: 0, timestamp: "" };
  }
}

// Main data loop
let frameCount = 0;
let advisorData: object = { recommendation: "Awaiting AI advisor...", reasons: [], confidence: 0, timestamp: "" };
let advisorCounter = 0;
async function dataLoop() {
  while (true) {
    frameCount++;
    advisorCounter++;
    if (advisorCounter % 30 === 0) {
      advisorData = getAdvisorRecommendation();
}
    // Capture frame
    const frame = captureFrame();
    
    // Get telemetry every cycle
    const telemetry = getTelemetry();

    // Get AI analysis every 5 frames
    let analysis = null;
    if (frameCount % 5 === 0 && frame) {
      analysis = await getAgentAnalysis(frame);
    }

    // Broadcast to dashboard
    broadcast({
      type: "update",
      frame: frame,
      telemetry,
      analysis,
      advisor: advisorData,
      timestamp: Date.now()
    });

    await new Promise(r => setTimeout(r, 1000));
  }
}

// WebSocket server
const server = serve({
  port: 3002,
  fetch(req, server) {
    // Serve dashboard HTML
    if (req.url.endsWith("/")) {
      return new Response(fs.readFileSync("./dashboard.html"), {
        headers: { "Content-Type": "text/html" }
      });
    }
    
    // Upgrade to WebSocket
    if (server.upgrade(req)) return;
    return new Response("Not found", { status: 404 });
  },
  websocket: {
    open(ws) {
      clients.add(ws);
      console.log(`Client connected (${clients.size} total)`);
    },
    close(ws) {
      clients.delete(ws);
      console.log(`Client disconnected (${clients.size} total)`);
    },
    message(ws, msg) {
      // Handle commands from dashboard
      try {
        const cmd = JSON.parse(msg as string);
        console.log(`[DASHBOARD CMD] ${JSON.stringify(cmd)}`);
        // Forward to drone.py via file or socket
      } catch {}
    }
  }
});

console.log("🚁 ARIA Server running on http://localhost:3002");
console.log("📡 WebSocket ready for dashboard connection");

dataLoop();