const $ = id => document.getElementById(id);
const names = {language:'QWEN / ENGLISH DECISIONS',local:'TINY IMITATION BASELINE',rules:'RULE-BASED TEACHER',random:'RANDOM BASELINE',manual:'KEYBOARD CONTROL',jev:'TYPESAFE JEV API'};
const descriptions = {hunt:'Track enemies. Fire when the target is lined up.',conserve:'Wait for a more precise shot before spending ammo.',pacifist:'Track targets without firing. Enforced by the controller.'};
let latest = null, busy = false, instructionDirty = false;
$('instruction').addEventListener('input', () => {instructionDirty = true;});
for (const name of ['left','hold','right']) {
  const row = document.createElement('div'); row.className = 'bar-row'; row.id = `row-${name}`;
  row.innerHTML = `<span>${name.toUpperCase()}</span><div class="bar-track"><i id="bar-${name}"></i></div><span id="p-${name}">0%</span>`;
  $('bars').append(row);
}
async function control(data) {
  try {
    const response = await fetch('/api/control', {method:'POST',headers:{'Content-Type':'application/json','X-OpenJev':'1'},body:JSON.stringify(data)});
    if (!response.ok) { const error = await response.json(); throw Error(typeof error.detail === 'string' ? error.detail : 'Invalid control settings'); }
    $('error').hidden = true;
    return true;
  } catch (e) { $('error').textContent = e.message; $('error').hidden = false; return false; }
}
$('toggle').onclick = () => control({command:latest?.status === 'running'?'pause':'start'});
$('restart').onclick = () => control({command:'restart',seed:Number($('seed').value)});
$('policy').onchange = () => control({command:'configure',policy:$('policy').value});
$('apply-instruction').onclick = async () => {if(await control({command:'configure',instruction:$('instruction').value})) instructionDirty = false;};
$('scenario').onchange = () => control({command:'configure',scenario:$('scenario').value});
$('seed').onchange = () => control({command:'configure',seed:Number($('seed').value)});
document.querySelectorAll('[data-directive]').forEach(b => b.onclick = () => control({command:'configure',directive:b.dataset.directive}));
async function poll() {
  if (busy) return;
  busy = true;
  try {
    const response = await fetch('/api/state');
    if (!response.ok) throw Error('Could not reach the game server');
    const s = await response.json(); latest = s;
    $('status').textContent = s.status.toUpperCase();
    $('live-dot').className = 'live-dot ' + s.status;
    $('toggle').textContent = s.status === 'running' ? 'Ⅱ PAUSE AGENT' : s.status === 'finished' ? '▶ NEW EPISODE' : '▶ RUN AGENT';
    if (s.frame) { $('game').src = 'data:image/jpeg;base64,' + s.frame; $('loading').hidden = true; }
    if (s.config) {
      for (const id of ['policy','scenario','seed']) if (document.activeElement !== $(id)) $(id).value = s.config[id];
      $('scenario-label').textContent = s.config.scenario.replaceAll('_',' ').toUpperCase();
      $('feed-policy').textContent = names[s.config.policy];
      $('backend-chip').textContent = s.config.policy.toUpperCase();
      $('language-settings').hidden = s.config.policy !== 'language';
      if (document.activeElement !== $('instruction') && !instructionDirty) $('instruction').value = s.config.instruction;
      const detail = {
        language:['English decision model','Context + instruction + six actions','Qwen candidate scoring. No generated explanation.'],
        local:['One forward pass','11 features → 64 → 64 → 5 logits','No generated text. Two output heads.'],
        rules:['Hand-written rules','Aim and fire thresholds','Deterministic teacher, no neural model.'],
        random:['Random controls','Uniform steering + coin-flip firing','Baseline with the same action space.'],
        manual:['Your keyboard','Left / right / space','Human controls with enforced constraints.'],
        jev:['TypeSafe API','Choice + Noul in one request','Remote call. Up to 300 attempts per server.']
      }[s.config.policy];
      ['pipeline-title','pipeline-detail','pipeline-note'].forEach((id,i) => $(id).textContent = detail[i]);
      $('directive-description').textContent = descriptions[s.config.directive];
      $('manual-help').hidden = s.config.policy !== 'manual';
      document.querySelectorAll('[data-directive]').forEach(b => b.classList.toggle('selected',b.dataset.directive === s.config.directive));
    }
    if (s.stats) { $('kills').textContent = s.stats.kills; $('time').textContent = s.stats.game_seconds.toFixed(1) + 's'; $('health').textContent = s.stats.health + '%'; $('ammo').textContent = s.stats.ammo; }
    if (s.observation) {
      const o = s.observation;
      $('target').textContent = o.visible ? `${o.target} / aim ${o.aim_error.toFixed(2)}` : 'No enemy visible / scanning';
    }
    if (s.decision) {
      const d = s.decision;
      $('latency').textContent = d.latency_ms.toFixed(2); $('steer').textContent = d.steer.toUpperCase();
      for (const name of ['left','hold','right']) {
        const percent = 100 * d.probabilities[name];
        $(`bar-${name}`).style.width = percent + '%'; $(`p-${name}`).textContent = Math.round(percent) + '%';
        $(`row-${name}`).classList.toggle('active',d.steer === name);
      }
      $('fire').textContent = Math.round(100*d.fire_probability) + '%'; $('fire-bar').style.width = 100*d.fire_probability+'%';
      const fire = d.fire && s.config.directive !== 'pacifist' && s.observation?.ammo > 0;
      $('action').textContent = `steer: ${d.steer}   fire: ${fire}`;
    } else {
      $('latency').textContent = '--'; $('steer').textContent = 'PENDING';
      $('fire').textContent = '--'; $('fire-bar').style.width = '0%';
      for (const name of ['left','hold','right']) {
        $(`bar-${name}`).style.width = '0%'; $(`p-${name}`).textContent = '--';
        $(`row-${name}`).classList.remove('active');
      }
      $('action').textContent = 'Waiting for this controller to act';
    }
    $('decisions').textContent = (s.decisions || 0).toLocaleString() + ' decisions';
    $('jev-option').disabled = !s.jev_available;
    $('jev-option').textContent = s.jev_available ? `Jev API / ${s.jev_calls}/300 calls` : 'Jev API / key required';
    if (s.error) { $('error').textContent = s.error; $('error').hidden = false; }
  } catch (e) { $('status').textContent = 'DISCONNECTED'; $('error').textContent = e.message; $('error').hidden = false; }
  finally { busy = false; }
}
const keys = new Set();
window.addEventListener('keydown', e => {
  if (latest?.config?.policy !== 'manual' || ['INPUT','SELECT','TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (['ArrowLeft','ArrowRight',' '].includes(e.key)) { e.preventDefault(); keys.add(e.key); }
});
window.addEventListener('keyup', e => keys.delete(e.key));
window.addEventListener('blur', () => keys.clear());
setInterval(() => {
  if (latest?.config?.policy === 'manual' && latest.status === 'running')
    control({command:'manual',steer:keys.has('ArrowLeft')?'left':keys.has('ArrowRight')?'right':'hold',fire:keys.has(' ')});
},100);
setInterval(poll,100); poll();
