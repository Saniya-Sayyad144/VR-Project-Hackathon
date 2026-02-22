async function fetchData(){
  const statusEl = document.getElementById('status');
  const exerciseTimeEl = document.getElementById('total_exercise_time');
  const vrTimeEl = document.getElementById('total_vr_time');
  const repsEl = document.getElementById('total_reps');
  const fatigueEl = document.getElementById('avg_fatigue');
  const tbody = document.getElementById('sessionsTable');

  // Safety check for required elements
  if(!exerciseTimeEl || !vrTimeEl || !repsEl || !fatigueEl || !tbody) {
    console.error('Missing required dashboard elements');
    if(statusEl) statusEl.innerText = 'Dashboard error: missing elements';
    return;
  }

  if(statusEl) statusEl.innerText = 'Loading...';
  
  try{
    const res = await fetch('/api/dashboard-data',{credentials:'include'})
    if(res.status===401){ window.location='/login-page'; return }
    const data = await res.json()
    
    if(data.error){
      exerciseTimeEl.innerText = '0s';
      vrTimeEl.innerText = '0 min';
      repsEl.innerText = '0';
      fatigueEl.innerText = 'Processing';
      tbody.innerHTML = '<tr><td colspan="5">No sessions yet</td></tr>';
      if(statusEl) statusEl.innerText = 'No data';
      return;
    }

    // Update stats
    exerciseTimeEl.innerText = formatSeconds(data.total_exercise_time || 0);
    vrTimeEl.innerText = formatMinutes(data.total_vr_time || 0);
    repsEl.innerText = data.total_reps || 0;
    fatigueEl.innerText = data.avg_fatigue===null || data.avg_fatigue===undefined ? 'Processing' : Number(data.avg_fatigue).toFixed(2);

    // Update sessions table
    tbody.innerHTML = '';
    if(!data.sessions || data.sessions.length===0){
      tbody.innerHTML = '<tr><td colspan="5">No sessions yet</td></tr>';
    }else{
      for(const s of data.sessions){
        const tr = document.createElement('tr');
        const dateStr = s.date ? new Date(s.date).toLocaleDateString() : '-';
        tr.innerHTML = `<td>${dateStr}</td><td>${s.exercise_name || '-'}</td><td>${s.duration || 0}</td><td>${s.reps || 0}</td><td>${s.fatigue===null || s.fatigue===undefined ? 'Processing' : s.fatigue}</td>`;
        tbody.appendChild(tr);
      }
    }
    if(statusEl) statusEl.innerText = 'Updated';
  }catch(e){
    console.error('Fetch error:', e);
    exerciseTimeEl.innerText = '0s';
    vrTimeEl.innerText = '0 min';
    repsEl.innerText = '0';
    fatigueEl.innerText = 'Processing';
    tbody.innerHTML = '<tr><td colspan="5">Error loading sessions</td></tr>';
    if(statusEl) statusEl.innerText = 'Error: ' + e.message;
  }
}

function formatSeconds(s){ if(!s || s<=0) return '0s'; if(s<60) return Math.round(s)+'s'; const m=Math.floor(s/60); const sec=Math.round(s%60); return `${m}m ${sec}s` }
function formatMinutes(m){ if(!m || m<=0) return '0 min'; return Math.round(m) + ' min' }
function setStatus(t){ 
  const statusEl = document.getElementById('status');
  if(statusEl) statusEl.innerText = t;
}


const refreshBtn = document.getElementById('refreshBtn');
if(refreshBtn) refreshBtn.addEventListener('click', ()=>{ fetchData() });

const downloadBtn = document.getElementById('downloadBtn');
if(downloadBtn) downloadBtn.addEventListener('click', async ()=>{
  setStatus('Preparing PDF...')
  try{
    const res = await fetch('/api/export/today',{credentials:'include'})
    if(res.status===401){ window.location='/login-page'; return }
    if(!res.ok){ const j=await res.json().catch(()=>null); setStatus(j && j.error? j.error : 'Export failed'); return }
    const blob = await res.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a'); a.href = url; a.download = 'today_report.pdf'; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(url)
    setStatus('Downloaded')
  }catch(e){ setStatus('Export error') }
});

const analyzeBtn = document.getElementById('analyzeBtn');
if(analyzeBtn) analyzeBtn.addEventListener('click', async ()=>{
  const modal = document.getElementById('aiModal'); const result = document.getElementById('aiResult');
  if(modal && result){
    modal.style.display='flex'; result.innerText = 'Analyzing...'
    try{
      const res = await fetch('/api/ai/day-analysis',{method:'POST',credentials:'include'})
      if(res.status===401){ window.location='/login-page'; return }
      const j = await res.json()
      if(j.analysis) result.innerText = j.analysis
      else result.innerText = j.error || 'Analysis unavailable'
    }catch(e){ result.innerText = 'Error contacting AI' }
  }
});

const closeAi = document.getElementById('closeAi');
if(closeAi) closeAi.addEventListener('click', ()=>{ const aiModal = document.getElementById('aiModal'); if(aiModal) aiModal.style.display='none'; });
