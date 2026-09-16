(() => {
  const el = id => document.getElementById(id);
  const examples = {
    support: {context:'A customer says: I was charged twice for my subscription this month. Please refund the duplicate payment.', questions:[
      {id:'queue',question:'Which team should handle this request?',candidates:[{id:'billing',description:'Billing and payments'},{id:'technical',description:'Technical troubleshooting'},{id:'sales',description:'New product sales'}]},
      {id:'requested_action',question:'What does the customer want?',candidates:[{id:'refund',description:'Refund a duplicate charge'},{id:'cancel',description:'Cancel their subscription'},{id:'upgrade',description:'Upgrade their subscription'}]}
    ]},
    meeting: {context:'Maya proposed launching on Friday. Alex said testing would not be finished until Monday. They agreed to revisit the launch date after testing.',questions:[
      {id:'launch',question:'Was Friday confirmed as the launch date?',candidates:[{id:'confirmed',description:'Yes, Friday was confirmed'},{id:'unresolved',description:'No, the date remains undecided'}]},
      {id:'next_step',question:'What should happen before deciding the launch date?',candidates:[{id:'test',description:'Finish testing'},{id:'announce',description:'Announce the Friday launch'}]}
    ]},
    game: {context:'The enemy is centered in the crosshair. Health is 80. Ammo is 12. The player is instructed to avoid firing, even when a target is available.',questions:[
      {id:'action',question:'Which available action follows the instruction?',candidates:[{id:'fire',description:'Fire at the enemy'},{id:'hold',description:'Keep aiming without firing'}]}
    ]}
  };
  let result = null, serial = 0;
  function invalidate() {
    if (!result) return;
    result=null; el('export-response').disabled=true;
    el('decision-results').replaceChildren(); el('response-json').textContent='No result for these inputs yet.';
    el('decision-status').textContent='Inputs changed. Score again for a fresh result.';
  }
  function node(tag, text, className) {
    const n = document.createElement(tag);
    if (text !== undefined) n.textContent = text;
    if (className) n.className = className;
    return n;
  }
  function field(label, value, className, max) {
    const wrap = node('label',label); const input = node('input');
    input.value = value; input.className = className; input.maxLength = max;
    wrap.append(input); return wrap;
  }
  function candidate(container, c={id:'',description:''}) {
    if (container.children.length >= 12) return;
    const row = node('div',undefined,'candidate-editor');
    row.append(field('ID',c.id,'candidate-id',64),field('CANDIDATE ANSWER',c.description,'candidate-description',1000));
    const remove = node('button','×','remove'); remove.setAttribute('aria-label','Remove candidate');
    remove.onclick = () => {if (container.children.length > 2) {row.remove(); invalidate();}};
    row.append(remove); container.append(row); invalidate();
  }
  function question(q) {
    if (el('questions').children.length >= 4) return;
    serial++;
    q ||= {id:`question_${serial}`,question:'',candidates:[{id:'yes',description:'Yes'},{id:'no',description:'No'}]};
    const card = node('div',undefined,'question-editor');
    const heading = node('div',undefined,'question-heading');
    heading.append(field('QUESTION ID',q.id,'question-id',64));
    const remove = node('button','REMOVE'); remove.onclick = () => {if(el('questions').children.length>1) {card.remove(); invalidate();}};
    heading.append(remove); card.append(heading,field('YOUR QUESTION',q.question,'question-text',2000));
    const choices = node('div',undefined,'candidate-list'); q.candidates.forEach(c => candidate(choices,c));
    const add = node('button','+ ADD CANDIDATE'); add.onclick = () => candidate(choices);
    card.append(choices,add); el('questions').append(card);
  }
  function load(name) {
    invalidate();
    el('context').value = examples[name].context; el('questions').replaceChildren(); serial=0;
    examples[name].questions.forEach(question);
  }
  function request() {
    return {context:el('context').value,questions:[...el('questions').children].map(q => ({
      id:q.querySelector('.question-id').value,question:q.querySelector('.question-text').value,
      candidates:[...q.querySelector('.candidate-list').children].map(c => ({id:c.querySelector('.candidate-id').value,description:c.querySelector('.candidate-description').value}))
    }))};
  }
  function download(data,name) {
    const url = URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
    const a=node('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function render(response,payload) {
    el('decision-results').replaceChildren();
    for (const answer of response.answers) {
      const q=payload.questions.find(q=>q.id===answer.id); const card=node('div',undefined,'answer-card');
      card.append(node('h3',q.question),node('p',`SELECTED / ${answer.choice}`,'selected-answer'));
      for (const c of q.candidates) {
        const row=node('div',undefined,'answer-row');
        const name=node('div'); name.append(node('span',c.description),node('small',c.id));
        row.append(name,node('strong',`${(answer.probabilities[c.id]*100).toFixed(1)}%`));
        const track=node('div',undefined,'answer-track'); const bar=node('i'); bar.style.width=answer.probabilities[c.id]*100+'%'; track.append(bar);
        card.append(row,track);
      }
      const details=node('details'); details.append(node('summary','Scoring details'),node('p',`${answer.input_tokens} input tokens · ${Math.round(answer.latency_ms)} ms · ${answer.entropy_nats.toFixed(3)} entropy (nats) · ${(answer.candidate_token_mass*100).toFixed(2)}% vocabulary probability on candidate letters. Entropy and token mass are diagnostics, not calibrated confidence.`,'hint'));
      card.append(details); el('decision-results').append(card);
    }
    el('response-json').textContent=JSON.stringify(response,null,2);
  }
  el('run-decision').onclick=async()=>{
    const payload=request(); el('run-decision').disabled=true; el('decision-error').hidden=true;
    const editors=[...document.querySelectorAll('.request-panel input, .request-panel textarea, .request-panel select, .request-panel button')];
    editors.forEach(e=>{e.disabled=true;});
    el('decision-status').textContent='Scoring locally. First use may take longer while the model loads…';
    result=null; el('export-response').disabled=true; el('decision-results').replaceChildren();
    el('response-json').textContent='Waiting for this request.';
    try {
      const r=await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json','X-OpenJev':'1'},body:JSON.stringify(payload)});
      const data=await r.json();
      if(!r.ok) throw Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));
      result=data; render(data,payload); el('export-response').disabled=false;
      el('decision-status').textContent=`${data.answers.length} decisions · ${(data.latency_ms/1000).toFixed(2)} seconds · no generated explanation`;
    } catch(e) {el('decision-status').textContent='No decision returned.'; el('decision-error').textContent=e.message;el('decision-error').hidden=false;}
    finally {editors.forEach(e=>{e.disabled=false;}); modelStatus();}
  };
  el('context').addEventListener('input',invalidate);
  el('questions').addEventListener('input',invalidate);
  el('example').onchange=()=>load(el('example').value);
  el('add-question').onclick=()=>question();
  el('export-request').onclick=()=>download(request(),'openjev-request.json');
  el('export-response').onclick=()=>{if(result)download(result,'openjev-result.json');};
  for(const view of ['decisions','doom']) el(`tab-${view}`).onclick=()=>{
    for(const v of ['decisions','doom']) {
      el(v==='decisions'?'decision-view':'doom-view').hidden=v!==view;
      el(`tab-${v}`).classList.toggle('selected',v===view); el(`tab-${v}`).setAttribute('aria-pressed',String(v===view));
    }
  };
  async function modelStatus(){try{const r=await fetch('/api/model');const s=await r.json();el('model-state').textContent=`LOCAL QWEN / ${s.status.replaceAll('_',' ').toUpperCase()}`;}catch{el('model-state').textContent='MODEL STATUS UNAVAILABLE';}}
  load('support'); modelStatus(); setInterval(modelStatus,2000);
})();
