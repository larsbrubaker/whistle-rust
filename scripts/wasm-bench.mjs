// Runs a wasm32-wasip1 build of the `bench` example (crates/whistle-rust/examples/bench.rs)
// under Node's V8, the engine Chrome uses, single-threaded. Called by scripts/wasm-bench.sh.
//
//   node scripts/wasm-bench.mjs [--preopen GUEST=HOST]... bench.wasm [bench args...]
//
// The current directory is preopened as "." so the example's relative paths (models/...,
// test-data/clips/...) resolve as they do natively; each --preopen adds one more directory (the
// worktree's `models` symlink target, the temp dir holding long.wav). Exits with the module's
// exit code.
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { argv, cwd, env, exit } from "node:process";

const args = argv.slice(2);
const preopens = { ".": cwd() };
while (args[0] === "--preopen") {
  const [guest, host] = args[1].split("=");
  preopens[guest] = host;
  args.splice(0, 2);
}
const [wasmPath, ...rest] = args;
if (!wasmPath) {
  console.error("usage: wasm-bench.mjs [--preopen GUEST=HOST]... bench.wasm [args...]");
  exit(2);
}

const wasi = new WASI({
  version: "preview1",
  args: ["bench", ...rest],
  env,
  preopens,
  returnOnExit: true,
});
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
