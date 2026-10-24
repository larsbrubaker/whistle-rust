// Fetches whistle.cact for the transcription worker (src/worker.ts): from the Cache API when a
// checked copy is there (the second visit downloads nothing), else from each URL in turn (the
// copy deployed next to the page first, then Hugging Face), reporting progress. Every copy is
// checked against the size and SHA-256 the library was tested with (whistle::model_file),
// with Web Crypto's SubtleCrypto.digest, before it is cached or used.

const CACHE_NAME = "whistle-model-v1";

export interface Progress {
  loaded: number;
  total: number;
  source: string;
}

async function sha256Hex(bytes: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function verified(bytes: Uint8Array<ArrayBuffer>, size: number, sha256: string): Promise<boolean> {
  return bytes.length === size && (await sha256Hex(bytes)) === sha256;
}

async function openCache(): Promise<Cache | null> {
  try {
    return typeof caches === "undefined" ? null : await caches.open(CACHE_NAME);
  } catch {
    // Private browsing or storage disabled: download every visit.
    return null;
  }
}

async function download(
  url: string,
  size: number,
  onProgress: (p: Progress) => void,
): Promise<Uint8Array<ArrayBuffer>> {
  const response = await fetch(url);
  if (!response.ok || !response.body) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  const total = Number(response.headers.get("content-length")) || size;
  const bytes = new Uint8Array(Math.max(total, size));
  const reader = response.body.getReader();
  let loaded = 0;
  let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > bytes.length) {
      throw new Error(`${url} is larger than whistle.cact`);
    }
    bytes.set(value, loaded);
    loaded += value.length;
    const now = performance.now();
    if (now - last > 100) {
      last = now;
      onProgress({ loaded, total, source: url });
    }
  }
  onProgress({ loaded, total, source: url });
  return bytes.subarray(0, loaded);
}

/** The checked model bytes and where they came from ("cache" or the URL). */
export async function loadModel(
  urls: string[],
  size: number,
  sha256: string,
  onProgress: (p: Progress) => void,
): Promise<{ bytes: Uint8Array<ArrayBuffer>; source: string }> {
  const cache = await openCache();
  // The cache is keyed by the checksum, so a new model version never reads an old copy.
  const key = `https://whistle-model.invalid/${sha256}`;
  if (cache) {
    const hit = await cache.match(key);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      if (await verified(bytes, size, sha256)) {
        onProgress({ loaded: size, total: size, source: "cache" });
        return { bytes, source: "cache" };
      }
      await cache.delete(key);
    }
  }
  const failures: string[] = [];
  for (const url of urls) {
    try {
      const bytes = await download(url, size, onProgress);
      if (!(await verified(bytes, size, sha256))) {
        failures.push(`${url}: not the expected whistle.cact (size or SHA-256 differs)`);
        continue;
      }
      if (cache) {
        try {
          await cache.put(key, new Response(bytes));
        } catch {
          // Over quota: still usable this visit.
        }
      }
      return { bytes, source: url };
    } catch (err) {
      failures.push(err instanceof Error ? err.message : String(err));
    }
  }
  throw new Error(`could not download the model: ${failures.join("; ")}`);
}
