// Microphone capture for the whistle-rust demo (used by src/main.ts): getUserMedia, then an
// AudioWorklet (public/capture-worklet.js) or, where there is none, a ScriptProcessor, giving
// mono f32 chunks of ~250 ms. The AudioContext asks for 16 kHz, which most browsers resample
// to; where that fails (Firefox will not connect a 48 kHz microphone to a 16 kHz context) it
// runs at the device rate and the worker resamples (whistle::audio::resample).

/** A running capture; `stop` releases the microphone. */
export interface Capture {
  rate: number;
  stop(): void;
}

const CHUNK_SECONDS = 0.25;

async function context(stream: MediaStream): Promise<{ ctx: AudioContext; source: MediaStreamAudioSourceNode }> {
  try {
    const ctx = new AudioContext({ sampleRate: 16000 });
    try {
      return { ctx, source: ctx.createMediaStreamSource(stream) };
    } catch {
      await ctx.close();
    }
  } catch {
    // No 16 kHz context here: fall through to the device rate.
  }
  const ctx = new AudioContext();
  return { ctx, source: ctx.createMediaStreamSource(stream) };
}

/** Ask for the microphone and start sending chunks to `onChunk`; throws when denied. */
export async function startCapture(onChunk: (samples: Float32Array, rate: number) => void): Promise<Capture> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("this browser has no microphone access (it needs HTTPS)");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  const { ctx, source } = await context(stream);
  const rate = ctx.sampleRate;
  const chunkLength = Math.round(rate * CHUNK_SECONDS);
  let buffer = new Float32Array(chunkLength);
  let filled = 0;
  const collect = (samples: Float32Array): void => {
    let offset = 0;
    while (offset < samples.length) {
      const take = Math.min(chunkLength - filled, samples.length - offset);
      buffer.set(samples.subarray(offset, offset + take), filled);
      filled += take;
      offset += take;
      if (filled === chunkLength) {
        onChunk(buffer, rate);
        buffer = new Float32Array(chunkLength);
        filled = 0;
      }
    }
  };
  let node: AudioNode;
  if (ctx.audioWorklet) {
    await ctx.audioWorklet.addModule(new URL("capture-worklet.js", document.baseURI).href);
    const worklet = new AudioWorkletNode(ctx, "whistle-capture");
    worklet.port.onmessage = (event: MessageEvent<Float32Array>) => collect(event.data);
    node = worklet;
  } else {
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (event) => collect(new Float32Array(event.inputBuffer.getChannelData(0)));
    node = processor;
  }
  source.connect(node);
  // Some engines only pull a node that reaches the destination; it outputs silence.
  node.connect(ctx.destination);
  await ctx.resume();
  return {
    rate,
    stop() {
      if (filled > 0) onChunk(buffer.slice(0, filled), rate);
      source.disconnect();
      node.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void ctx.close();
    },
  };
}
