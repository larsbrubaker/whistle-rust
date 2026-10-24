// AudioWorklet processor for the whistle-rust demo's microphone (loaded by src/microphone.ts):
// posts each render quantum's first channel (mono f32, at the AudioContext's rate) to the main
// thread, which batches ~250 ms and sends it to the transcription worker. Kept as plain JS in
// public/ because worklets load by URL.
class WhistleCapture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length > 0) {
      this.port.postMessage(channel.slice(0));
    }
    return true;
  }
}

registerProcessor("whistle-capture", WhistleCapture);
