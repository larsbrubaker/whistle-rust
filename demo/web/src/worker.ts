// The whistle-rust demo's transcription Web Worker: one thread that owns the model, so the
// page's UI thread never runs it. It loads the worker wasm module (pkg/worker/, built by
// build-wasm.sh with simd128) and uses its `Engine` export. Messages (see src/protocol.ts):
//   in:  init {urls} | start {options} | push {samples, rate} | finish | file {name, bytes, options}
//   out: progress {loaded, total, source} | loading | ready {source, ms} |
//        update {text, continues, pending, pendingContinues, language, ms, audioS} |
//        final {text, language, words, ms, audioS} | error {message}
// The model download, check and cache live in src/model.ts.
import init, { Engine, model_bytes, model_sha256, model_url } from "../pkg/worker/whistle_demo_worker.js";
import { loadModel } from "./model";
import type { Options, ToWorker } from "./protocol";

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let engine: Engine | null = null;

function post(message: object): void {
  ctx.postMessage(message);
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function startStream(options: Options): void {
  engine!.start(options.language, options.beams, options.keywords, options.wordTimestamps);
}

async function handle(msg: ToWorker): Promise<void> {
  switch (msg.type) {
    case "init": {
      await init();
      const urls = [...msg.urls, model_url()].filter((u, i, all) => u && all.indexOf(u) === i);
      const { bytes, source } = await loadModel(urls, model_bytes(), model_sha256(), (p) =>
        post({ type: "progress", ...p }),
      );
      post({ type: "loading" });
      const t0 = performance.now();
      engine = new Engine(bytes);
      post({ type: "ready", source, ms: performance.now() - t0 });
      return;
    }
    case "start":
      startStream(msg.options);
      return;
    case "push": {
      const t0 = performance.now();
      const update = JSON.parse(engine!.push(msg.samples, msg.rate));
      post({
        type: "update",
        ...update,
        ms: performance.now() - t0,
        audioS: msg.samples.length / msg.rate,
      });
      return;
    }
    case "finish": {
      const t0 = performance.now();
      const last = JSON.parse(engine!.finish());
      post({
        type: "final",
        text: last.full,
        language: last.language,
        words: last.words,
        ms: performance.now() - t0,
        audioS: 0,
      });
      return;
    }
    case "file": {
      const o = msg.options;
      const audioS = Engine.wav_seconds(msg.bytes);
      const t0 = performance.now();
      const result = JSON.parse(
        engine!.transcribe_wav(msg.bytes, o.language, o.beams, o.keywords, o.wordTimestamps),
      );
      post({ type: "final", ...result, ms: performance.now() - t0, audioS });
      return;
    }
  }
}

// One message at a time, in order: a push waits for the one before it.
let queue = Promise.resolve();
ctx.onmessage = (event: MessageEvent<ToWorker>) => {
  queue = queue.then(() =>
    handle(event.data).catch((err) => post({ type: "error", message: errorText(err) })),
  );
};
