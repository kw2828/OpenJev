import {CreateMLCEngine, prebuiltAppConfig} from 'https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/lib/index.js';
import {MODEL_ID, MODEL_REVISION, buildMessages, scoreLogits, validateRequest} from './decision-core.mjs';
let engine = null, busy = false, activeCandidates = null, captured = null;
const processor = {
  processLogits(logits) {
    if (activeCandidates) captured = scoreLogits(logits, activeCandidates);
    return logits;
  },
  processSampledToken() {},
  resetState() { captured = null; }
};
self.onmessage = async ({data}) => {
  if (busy) return self.postMessage({type:'error',message:'A model request is already running.'});
  busy = true;
  try {
    if (data.type === 'load') {
      if (engine) throw new Error('Model is already loaded.');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter || !adapter.features.has('shader-f16')) throw new Error('This model requires a WebGPU adapter with shader-f16 support.');
      const config = prebuiltAppConfig.model_list.find(m=>m.model_id===MODEL_ID);
      if (!config) throw new Error('Pinned model is missing from the runtime catalog.');
      const started = performance.now();
      engine = await CreateMLCEngine(MODEL_ID, {
        appConfig:{model_list:[{...config,
          model:`https://huggingface.co/mlc-ai/${MODEL_ID}/resolve/${MODEL_REVISION}/`} ]},
        logitProcessorRegistry:new Map([[MODEL_ID,processor]]),
        initProgressCallback: r=>self.postMessage({type:'progress',text:r.text,progress:r.progress})
      });
      self.postMessage({type:'loaded',load_ms:performance.now()-started,model:MODEL_ID,revision:MODEL_REVISION});
    } else if (data.type === 'score') {
      if (!engine) throw new Error('Load the model first.');
      const request = validateRequest(data.request);
      await engine.resetChat();
      activeCandidates = request.candidates;
      captured = null;
      const started = performance.now();
      const completion = await engine.chat.completions.create({
        messages:buildMessages(request),max_tokens:1,temperature:1,top_p:1,
        frequency_penalty:0,presence_penalty:0,extra_body:{enable_thinking:false}
      });
      if (!captured) throw new Error('The runtime did not expose candidate logits. No scores were produced.');
      self.postMessage({type:'result',result:{...captured,elapsed_ms:performance.now()-started,
        model:MODEL_ID,revision:MODEL_REVISION,runtime:'WebLLM 0.2.85 / WebGPU',
        scoring:'raw first-position logits normalized over verified A-H tokens; one token sampled and discarded',
        usage:completion.usage,request}});
      activeCandidates = null;
    }
  } catch (error) {
    activeCandidates = null;
    self.postMessage({type:'error',message:error.message || String(error)});
  } finally { busy = false; }
};
