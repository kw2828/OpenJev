import {MODEL_ID, MODEL_REVISION, buildMessages, scoreLogits, validateRequest} from './decision-core.mjs';
const fetchAsset = self.fetch.bind(self);
self.fetch = async (input, init) => {
  const url = typeof input === 'string' || input instanceof URL ? String(input) : input.url;
  const name = new URL(url, self.location.href).pathname.split('/').pop();
  self.postMessage({type:'progress',text:`Downloading ${name}…`});
  try { return await fetchAsset(input, init); }
  catch (error) { throw new Error(`Could not download ${name}: ${error.message}`); }
};
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
      self.postMessage({type:'progress',text:'Checking GPU support in the model worker…'});
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter || !adapter.features.has('shader-f16')) throw new Error('This model requires a WebGPU adapter with shader-f16 support.');
      self.postMessage({type:'progress',text:'Downloading the WebLLM runtime…'});
      const {CreateMLCEngine, prebuiltAppConfig} = await import('https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/lib/index.js');
      self.postMessage({type:'progress',text:'Loading model configuration and GPU library…'});
      const config = prebuiltAppConfig.model_list.find(m=>m.model_id===MODEL_ID);
      if (!config) throw new Error('Pinned model is missing from the runtime catalog.');
      const started = performance.now();
      engine = await CreateMLCEngine(MODEL_ID, {
        appConfig:{cacheBackend:'indexeddb',model_list:[{...config,
          model_lib:'https://cdn.jsdelivr.net/gh/mlc-ai/binary-mlc-llm-libs@025bcaf3780fa8254f5e5efd3bfea0a5397248f4/web-llm-models/v0_2_84/base/Qwen3-0.6B-q4f16_1_cs1k-webgpu.wasm',
          integrity:{model_lib:'sha256-TbgAskEZIE4aA4booS4ITVASqmD3fFv/rTYvIEmN+RI=',onFailure:'error'},
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
