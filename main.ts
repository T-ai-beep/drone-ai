import Ollama from "ollama";
import { execSync, spawn } from "child_process";
import * as fs from "fs";
import { interpretCommand } from "./interperter";

const FRAME_PATH = "/Users/tanayshah/drone/snapshot.jpg";
const AUDIO_PATH = "/tmp/drone_voice.wav";

async function captureFrame(): Promise<string> {
  execSync(`ffmpeg -f avfoundation -framerate 30 -video_size 1280x720 -i "0" -vframes 1 -update 1 ${FRAME_PATH} -y 2>/dev/null`);
  return fs.readFileSync(FRAME_PATH).toString("base64");
}

async function analyzeScene(base64Image: string, command: string): Promise<string> {
  const response = await Ollama.chat({
    model: "llava",
    messages: [{
      role: "user",
      content: `You are an aerial drone AI. The pilot said: "${command}". Looking at the scene below, respond with what action you would take and what you observe. Be concise.`,
      images: [base64Image],
    }],
  });
  return response.message.content;
}

async function recordAndTranscribe(): Promise<string> {
  await new Promise<void>((resolve, reject) => {
    const rec = spawn("sox", [
      "-t", "coreaudio", "default",
      "-r", "24000", "-c", "1", "-b", "16", "-e", "signed-integer",
      AUDIO_PATH, "trim", "0", "3"
    ]);
    rec.on("close", resolve);
    rec.on("error", reject);
  });

  const result = execSync(`python3 -c "
import whisper
model = whisper.load_model('base')
result = model.transcribe('${AUDIO_PATH}')
print(result['text'])
"`);
  return result.toString().trim();
}

async function main() {
  console.log("🚁 DRONE AI SYSTEM ONLINE");
  console.log("Speak a command after the listening prompt...\n");

  while (true) {
    console.log("🎤 Listening for command...");
    const command = await recordAndTranscribe();
    
    if (!command) {
      console.log("No command detected, listening again...\n");
      continue;
    }

    console.log(`[PILOT] ${command}`);
    const droneCommand = await interpretCommand(command);
    console.log(`[COMMAND] ${JSON.stringify(droneCommand)}`);
    console.log("📸 Capturing scene...");
    
    const frame = await captureFrame();
    
    console.log("🧠 AI analyzing...");
    const response = await analyzeScene(frame, command);
    
    console.log(`[DRONE AI] ${response}\n`);
  }
}

main();