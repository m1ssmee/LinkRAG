// The session travels as a header (and ?s= on media URLs), not a cookie: a Space runs in a
// cross-site iframe, where cookies are not sent. One session per tab.
function sessionId() {
  const key = 'lectern-session'
  try {
    const kept = sessionStorage.getItem(key)
    if (kept) return kept
  } catch {
    /* storage blocked: a fresh session per load */
  }
  const id = Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('')
  try {
    sessionStorage.setItem(key, id)
  } catch {
    /* as above */
  }
  return id
}

const SID = sessionId()

export const withSession = (url) => url && `${url}${url.includes('?') ? '&' : '?'}s=${SID}`

export async function api(path, { method = 'GET', json, body } = {}) {
  const headers = { 'X-Session': SID }
  if (json !== undefined) {
    headers['Content-Type'] = 'application/json'
    body = JSON.stringify(json)
  }
  const response = await fetch(path, { method, headers, body })
  if (!response.ok) {
    let detail = response.statusText
    try {
      detail = (await response.json()).detail ?? detail
    } catch {
      /* not JSON */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return response
}

export const getJSON = async (path) => (await api(path)).json()
export const sendJSON = async (path, method, json) => (await api(path, { method, json })).json()

export async function uploadFiles(files) {
  const form = new FormData()
  for (const file of files) form.append('files', file)
  return (await api('/upload', { method: 'POST', body: form })).json()
}

/** POST a job endpoint and call onEvent for each NDJSON line it streams. */
export async function streamJob(path, onEvent) {
  const reader = (await api(path, { method: 'POST' })).body.getReader()
  const decoder = new TextDecoder()
  let buffered = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffered += decoder.decode(value, { stream: true })
    let end
    while ((end = buffered.indexOf('\n')) >= 0) {
      const line = buffered.slice(0, end).trim()
      buffered = buffered.slice(end + 1)
      if (line) onEvent(JSON.parse(line))
    }
  }
}

export function clock(seconds) {
  const s = Math.max(0, Math.round(seconds || 0))
  const mm = String(Math.floor((s % 3600) / 60)).padStart(s >= 3600 ? 2 : 1, '0')
  return `${s >= 3600 ? `${Math.floor(s / 3600)}:` : ''}${mm}:${String(s % 60).padStart(2, '0')}`
}

export const LINK_TYPES = ['audio_slide', 'figure_text', 'deictic', 'same_slide']
