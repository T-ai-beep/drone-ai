import Ollama from "ollama";
import { execSync } from "child_process";
import * as fs from "fs";

const CAPTURE_PATH = "/Users/tanayshah/drone/snapshot.jpg";
async function captureFrame(): Promise<string> {
  // Capture frame from webcam using imagesnap (mac)
  execSync(`imagesnap -q ${CAPTURE_PATH}`);
  const imageData = fs.readFileSync(CAPTURE_PATH);
  return imageData.toString("base64");
}

async function analyzeFrame(base64Image: string): Promise<string> {
  const response = await Ollama.chat({
    model: "llava",
    messages: [
      {
        role: "user",
        content: "You are an aerial drone AI. Describe what you see below you. Focus on terrain, obstacles, and points of interest. Be concise.",
        images: [base64Image],
      },
    ],
  });
  return response.message.content;
}

async function visionLoop() {
  console.log("🚁 Drone vision system starting...");
  
  while (true) {
    try {
      const frame = await captureFrame();
      const analysis = await analyzeFrame(frame);
      console.log(`[VISION] ${analysis}\n`);
    } catch (err) {
      console.error("Frame error:", err);
    }
  }
}

visionLoop();