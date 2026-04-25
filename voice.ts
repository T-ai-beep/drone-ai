import { execSync, spawn } from "child_process";
import * as fs from "fs";

const RECORDING_PATH = "/tmp/drone_voice.wav";

async function recordAudio(seconds: number = 3): Promise<void> {
  return new Promise((resolve, reject) => {
    console.log("🎤 Listening...");
    const rec = spawn("sox", [
      "-t", "coreaudio",
      "default",
      "-r", "16000",
      "-c", "1",
      "-b", "16",
      RECORDING_PATH,
      "trim", "0", `${seconds}`
    ]);
    rec.on("close", resolve);
    rec.on("error", reject);
  });
}

async function transcribe(): Promise<string> {
  const result = execSync(
    `python3 -c "
import whisper, sys
model = whisper.load_model('base')
result = model.transcribe('${RECORDING_PATH}')
print(result['text'])
"`
  );
  return result.toString().trim();
}

async function voiceLoop() {
  console.log("🚁 Voice control system starting...");
  
  while (true) {
    await recordAudio(3);
    const command = await transcribe();
    console.log(`[COMMAND] ${command}`);
  }
}

voiceLoop();