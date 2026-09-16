import {validateRequest} from './decision-core.mjs';
const el = id=>document.getElementById(id);
let worker=null, ready=false, busy=false, result=null, timer=null;
const examples = {
  support:{context:'A customer was charged twice for one order. They want the duplicate charge refunded.',question:'Which team should handle this request?',candidates:[{id:'billing',description:'Billing and payments'},{id:'technical',description:'Technical troubleshooting'},{id:'shipping',description:'Shipping and delivery'}]},
  game:{context:'An enemy is visible near the crosshair. The player has no ammunition remaining.',question:'Which action follows the instruction to avoid firing with an empty weapon?',candidates:[{id:'hold_fire',description:'Hold fire and scan for targets'},{id:'fire',description:'Fire at the visible enemy'},{id:'uncertain',description:'Not enough information'}]},
  meeting:{context:'The team discussed a Friday launch, but the owner said the date remains tentative until testing finishes.',question:'Was the launch date confirmed?',candidates:[{id:'confirmed',description:'The date was confirmed'},{id:'tentative',description:'The date remains tentative'}]}
};
function status(text,error=false) { el('status').textContent=text; el('status').classList.toggle('error',error); }
function controls() {
  el('load').disabled=busy || ready;
  el('score').disabled=busy || !ready;
  el('cancel').hidden=!busy && !ready;
  el('cancel').textContent=busy?'Cancel and release model':'Release model';
  el('export').disabled=!result;
}
function addCandidate(c={id:'choice_'+(el('candidates').children.length+1),description:''}) {
  if (el('candidates').children.length>=8) return;
  const row=document.createElement('div'); row.className='candidate';
  const id=document.createElement('input'); id.value=c.id; id.maxLength=40; id.className='candidate-id'; id.setAttribute('aria-label','Candidate ID');
  const description=document.createElement('input'); description.value=c.description; description.maxLength=240; description.className='candidate-description'; description.setAttribute('aria-label','Candidate description');
  const remove=document.createElement('button'); remove.textContent='×'; remove.className='remove'; remove.setAttribute('aria-label','Remove candidate');
  remove.onclick=()=>{row.remove();};
  row.append(id,description,remove); el('candidates').append(row);
}
function example(key) {
  const e=examples[key]; el('context').value=e.context; el('question').value=e.question;
  el('candidates').replaceChildren(); e.candidates.forEach(addCandidate);
}
function request() {
  return validateRequest({context:el('context').value,question:el('question').value,
    candidates:[...el('candidates').children].map(r=>({id:r.querySelector('.candidate-id').value.trim(),description:r.querySelector('.candidate-description').value.trim()}))});
}
function release(message='Model released. Downloaded weights may remain in your browser cache.') {
  worker?.terminate(); worker=null; ready=false; busy=false; clearTimeout(timer); controls(); status(message);
}
function showResult(data) {
  result=data; el('results').replaceChildren();
  const heading=document.createElement('h3'); heading.textContent='Selected: '+data.choice;
  const asked=document.createElement('p'); asked.className='muted'; asked.textContent=data.request.question;
  el('results').append(heading,asked);
  for (const c of data.probabilities) {
    const line=document.createElement('div'); line.className='probability';
    const label=document.createElement('span'); label.textContent=c.id+' · '+c.description;
    const value=document.createElement('strong'); value.textContent=(100*c.probability).toFixed(1)+'%';
    const bar=document.createElement('div'); bar.className='bar';
    const fill=document.createElement('div'); fill.style.width=(100*c.probability)+'%'; bar.append(fill);
    line.append(label,value,bar); el('results').append(line);
  }
  const meta=document.createElement('p'); meta.className='muted';
  meta.textContent=`${(data.elapsed_ms/1000).toFixed(2)} s in the model worker · candidate-label mass ${(100*data.candidate_label_mass).toFixed(3)}% · Qwen3 0.6B, 4-bit`;
  el('results').append(meta);
}
el('load').onclick=async()=>{
  if (!navigator.gpu) return status('This browser does not expose WebGPU. Try a recent desktop Chrome or Edge. Recorded gameplay below works without it.',true);
  busy=true; controls(); status('Checking your browser GPU…');
  try {
    worker=new Worker('./model-worker.mjs?v=2',{type:'module'});
    worker.onerror=e=>{release(); status('Model worker could not start: '+e.message,true);};
    worker.onmessage=({data})=>{
      if (data.type==='progress') status(data.text);
      if (data.type==='loaded') {ready=true;busy=false;clearTimeout(timer);status(`Ready. Model setup took ${(data.load_ms/1000).toFixed(1)} s. Enter a decision and score it.`);controls();}
      if (data.type==='result') {busy=false;clearTimeout(timer);showResult(data.result);status('Scored locally. Results refer to the request shown above; export includes the full input.');controls();}
      if (data.type==='error') {busy=false;clearTimeout(timer);if(!ready){worker.terminate();worker=null;}status(data.message,true);controls();}
    };
    worker.postMessage({type:'load'});
    timer=setTimeout(()=>release('Model setup timed out after 10 minutes. You can retry; cached weights may be reused.'),600000);
  } catch(e) {busy=false;status(e.message,true);controls();}
};
el('score').onclick=()=>{
  try {
    const input=request(); busy=true; result=null; el('results').textContent='Scoring the current request…';controls(); status('Running one model step on your GPU. First scoring can include additional compilation.');
    worker.postMessage({type:'score',request:input});
    timer=setTimeout(()=>release('Scoring timed out after 90 seconds. Load the model again to retry.'),90000);
  } catch(e) {status(e.message,true);}
};
el('cancel').onclick=()=>release();
el('add').onclick=()=>addCandidate();
el('example').onchange=()=>example(el('example').value);
el('export').onclick=()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));
  const a=document.createElement('a'); a.href=url; a.download='decisiontics-result.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
};
example('support'); controls();
if (!navigator.gpu) status('WebGPU is unavailable here. Try a recent desktop Chrome or Edge, or view the recorded Doom episodes below.',true);
