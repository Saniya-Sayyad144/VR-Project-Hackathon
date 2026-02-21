async function fetchData(){
  setStatus('Loading...')
  try{
    const res = await fetch('/api/dashboard-data',{credentials:'include'})
    if(res.status===401){ window.location='/login-page'; return }
    const data = await res.json()
    if(data.error){ setStatus('Error'); return }

    document.getElementById('total_exercise_time').innerText = formatSeconds(data.total_exercise_time)
    document.getElementById('total_vr_time').innerText = formatMinutes(data.total_vr_time)
    document.getElementById('total_reps').innerText = data.total_reps
    document.getElementById('avg_fatigue').innerText = data.avg_fatigue===null? 'Processing' : Number(data.avg_fatigue).toFixed(2)

    const tbody = document.getElementById('sessionsTable')
    tbody.innerHTML = ''
    if(!data.sessions || data.sessions.length===0){
      tbody.innerHTML = '<tr><td colspan="5">No sessions</td></tr>'
    }else{
      for(const s of data.sessions){
        const tr = document.createElement('tr')
        tr.innerHTML = `<td>${s.date||''}</td><td>${s.exercise_name||''}</td><td>${s.duration||0}</td><td>${s.reps||0}</td><td>${s.fatigue===null? 'Processing' : s.fatigue}</td>`
        tbody.appendChild(tr)
      }
    }
    setStatus('')
  }catch(e){ setStatus('Error fetching data') }
}

function formatSeconds(s){ if(!s) return '0s'; if(s<60) return s+'s'; const m=Math.floor(s/60); const sec=s%60; return `${m}m ${sec}s` }
function formatMinutes(m){ if(!m) return '0 min'; return m + ' min' }
function setStatus(t){ document.getElementById('status').innerText = t }

document.getElementById('refreshBtn').addEventListener('click', ()=>{ fetchData() })
document.getElementById('downloadBtn').addEventListener('click', async ()=>{
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
})

document.getElementById('analyzeBtn').addEventListener('click', async ()=>{
  const modal = document.getElementById('aiModal'); const result = document.getElementById('aiResult');
  modal.style.display='flex'; result.innerText = 'Analyzing...'
  try{
    const res = await fetch('/api/ai/day-analysis',{method:'POST',credentials:'include'})
    if(res.status===401){ window.location='/login-page'; return }
    const j = await res.json()
    if(j.analysis) result.innerText = j.analysis
    else result.innerText = j.error || 'Analysis unavailable'
  }catch(e){ result.innerText = 'Error contacting AI' }
})
document.getElementById('closeAi').addEventListener('click', ()=>{ document.getElementById('aiModal').style.display='none' })

fetchData()
