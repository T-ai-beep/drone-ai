import Ollama from "ollama";

export interface DroneCommand {
  command: string;
  direction?: string;
  duration?: number;
  radius?: number;
  speed?: number;
  raw: string;
}

export async function interpretCommand(voiceInput: string): Promise<DroneCommand> {
  const response = await Ollama.chat({
    model: "llama3.2",
    messages: [{
      role: "user",
      content: `You are a drone command interpreter. Convert this voice command into a JSON drone command.

Voice input: "${voiceInput}"

Valid commands: move, scan, orbit, hover, land, takeoff, return

Respond with ONLY valid JSON, nothing else. Example:
{"command": "scan", "direction": "left", "duration": 5}

JSON:`
    }],
  });

  try {
    const json = JSON.parse(response.message.content.trim());
    return { ...json, raw: voiceInput };
  } catch {
    return { command: "hover", raw: voiceInput };
  }
}