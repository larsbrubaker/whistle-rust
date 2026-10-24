// The message protocol between the page (src/main.ts) and the transcription worker
// (src/worker.ts). The worker's replies are plain objects, documented in worker.ts.

/** The UI's options, as the worker's Engine takes them. */
export interface Options {
  /** "" for auto-detect, else a language code ("en", "de", ...). */
  language: string;
  beams: number;
  keywords: string[];
  wordTimestamps: boolean;
}

export type ToWorker =
  | { type: "init"; urls: string[] }
  | { type: "start"; options: Options }
  | { type: "push"; samples: Float32Array; rate: number }
  | { type: "finish" }
  | { type: "file"; name: string; bytes: Uint8Array; options: Options };
