// The whistle-rust demo's page glue: starts the agg-gui UI (pkg/ui/, demo/whistle-demo-web),
// starts the transcription worker (src/worker.ts) and tells it which whistle.cact to load,
// and connects the two: the UI's calls arrive on `globalThis.whistleHost` (record, stop, pick a
// file, transcribe a dropped file, open a link), and the worker's replies go to the UI's
// `ui_*` exports. The model URL is `?model=URL`, else the canvas's data-model (a copy next to
// the page), with Hugging Face as the worker's fallback. `window.whistleDemo` is for page
// automation: `snapshot()` (the UI's state) and `transcribeUrl(url)` (as a dropped file).
import init, {
  start_ui,
  ui_download,
  ui_drop_file,
  ui_final,
  ui_phase,
  ui_snapshot,
  ui_update,
} from "../pkg/ui/whistle_demo_web.js";
import { startCapture, type Capture } from "./microphone";
import type { Options, ToWorker } from "./protocol";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let capture: Capture | null = null;

function send(message: ToWorker, transfer: Transferable[] = []): void {
  worker.postMessage(message, transfer);
}

function options(language: string, beams: number, keywords: string[], wordTimestamps: boolean): Options {
  return { language, beams, keywords, wordTimestamps };
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

interface Word {
  word: string;
  start: number;
  end: number;
}

worker.onmessage = (event: MessageEvent) => {
  const m = event.data;
  switch (m.type) {
    case "progress":
      ui_download(m.loaded, m.total, m.source);
      break;
    case "loading":
      ui_phase("loading", "");
      break;
    case "ready":
      console.log(`whistle: model from ${m.source}, built in ${m.ms.toFixed(0)} ms`);
      ui_phase("ready", "");
      break;
    case "update":
      ui_update(m.text, m.continues, m.pending, m.pendingContinues, m.language ?? undefined, m.ms, m.audioS);
      break;
    case "final": {
      const words: Word[] = m.words ?? [];
      const times = new Float32Array(words.flatMap((w) => [w.start, w.end]));
      ui_final(m.text, m.language ?? undefined, words.map((w) => w.word).join("\n"), times, m.ms, m.audioS);
      break;
    }
    case "error":
      if (capture) {
        capture.stop();
        capture = null;
      }
      ui_phase("error", m.message);
      break;
  }
};

const host = {
  async startRecording(language: string, beams: number, keywords: string[], stamps: boolean) {
    try {
      const started = await startCapture((samples, rate) =>
        send({ type: "push", samples, rate }, [samples.buffer]),
      );
      send({ type: "start", options: options(language, beams, keywords, stamps) });
      capture = started;
      ui_phase("listening", "");
    } catch (err) {
      const denied = err instanceof DOMException && err.name === "NotAllowedError";
      ui_phase(
        "error",
        denied
          ? "microphone permission was denied; allow it in the address bar and press Record again"
          : `could not start the microphone: ${errorText(err)}`,
      );
    }
  },
  stopRecording() {
    if (!capture) {
      ui_phase("ready", "");
      return;
    }
    capture.stop();
    capture = null;
    send({ type: "finish" });
  },
  // The UI passes its options; the picked file goes back through the UI (as a drop does),
  // which reads them again.
  pickFile() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".wav,audio/wav,audio/x-wav";
    input.onchange = async () => {
      const file = input.files && input.files[0];
      if (!file) return;
      const bytes = new Uint8Array(await file.arrayBuffer());
      ui_drop_file(file.name, bytes);
    };
    input.click();
  },
  transcribeFile(
    name: string,
    bytes: Uint8Array,
    language: string,
    beams: number,
    keywords: string[],
    stamps: boolean,
  ) {
    send({ type: "file", name, bytes, options: options(language, beams, keywords, stamps) }, [bytes.buffer]);
  },
  openUrl(url: string) {
    window.open(url, "_blank", "noopener");
  },
};

(globalThis as unknown as { whistleHost: typeof host }).whistleHost = host;
async function boot(): Promise<void> {
  await init();
  start_ui();
  // For page automation, once the module is up.
  (window as unknown as { whistleDemo: object }).whistleDemo = {
    snapshot: () => JSON.parse(ui_snapshot()),
    async transcribeUrl(url: string) {
      const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
      ui_drop_file(url.split("/").pop() ?? url, bytes);
    },
  };
  const canvas = document.getElementById("whistle-canvas");
  const param = new URLSearchParams(window.location.search).get("model");
  const configured = param ?? canvas?.dataset.model ?? "./whistle.cact";
  send({ type: "init", urls: [new URL(configured, document.baseURI).href] });
}

boot().catch((err) => {
  console.error(err);
  ui_phase("error", errorText(err));
});
