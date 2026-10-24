// Vite config for the whistle-rust browser demo (web/). GitHub Pages serves it at
// https://larsbrubaker.github.io/whistle-rust/, so asset paths are relative (`./`), which works
// there and under `vite dev`. The transcription worker (src/worker.ts) is an ES module worker
// that imports the worker wasm module (pkg/worker/); the page imports pkg/ui/.
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022" },
  server: { host: true },
  preview: { host: true },
});
