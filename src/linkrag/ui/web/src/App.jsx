import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import AnswerPane, { Composer } from './Answer.jsx'
import Evidence from './Evidence.jsx'
import Materials from './Materials.jsx'
import Player from './Player.jsx'
import { Building, Empty } from './Start.jsx'
import TopBar from './TopBar.jsx'
import WhySheet from './WhySheet.jsx'
import { getJSON, sendJSON, streamJob, uploadFiles } from './lib.js'
import { STRINGS } from './strings.js'

function useStored(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const kept = localStorage.getItem(key)
      return kept == null ? initial : JSON.parse(kept)
    } catch {
      return initial
    }
  })
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      /* a preference only */
    }
  }, [key, value])
  return [value, setValue]
}

const BUILD_STAGES = ['ingest', 'index', 'link', 'pairs']

/** Apply one streamed job event (see api.stream_job) to the building view's state. */
function applyEvent(job, e) {
  const next = { ...job, stages: { ...job.stages } }
  if (e.stage && e.status) {
    next.stages[e.stage] = e.status === 'running'
      ? { status: 'running', started: Date.now() }
      : { ...next.stages[e.stage], status: e.status, seconds: e.seconds }
    if (!next.order.includes(e.stage)) next.order = [...next.order, e.stage]
    if (e.status === 'running') next.log = null
  }
  if (e.log) next.log = e.log
  if (e.warning) next.warnings = [...next.warnings, e.warning]
  if (e.status === 'queued') next.queued = true
  if (e.status === 'error') Object.assign(next, { error: e.detail, done: true })
  if (e.status === 'ready') next.done = true
  return next
}

export default function App() {
  const [lang, setLang] = useStored('lectern-lang', 'en')
  const [compare, setCompare] = useStored('lectern-compare', false)
  const t = STRINGS[lang]
  const [state, setState] = useState(null)
  const [job, setJob] = useState(null)
  const [uploadError, setUploadError] = useState(null)
  const [pairs, setPairs] = useState({ pairs: [], pending: false })
  const [stats, setStats] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [qa, setQa] = useState(null)
  const [hover, setHover] = useState(null)
  const [focusN, setFocusN] = useState(null)
  const [why, setWhy] = useState(false)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef(null)
  const stopAt = useRef(null)

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  const refresh = useCallback(async () => {
    const next = await getJSON('/state')
    setState(next)
    if (next.corpus && !next.building) {
      const [p, s, tl] = await Promise.all([getJSON('/pairs'), getJSON('/stats'), getJSON('/timeline')])
      setPairs(p)
      setStats(s)
      setTimeline(tl)
    }
    return next
  }, [])
  useEffect(() => {
    refresh()
  }, [refresh])

  // A build outlives a reload: the job keeps running on the server, so wait for it here.
  useEffect(() => {
    if (!state?.building || job) return undefined
    const timer = setInterval(async () => {
      if (!(await getJSON('/state')).building) refresh()
    }, 2000)
    return () => clearInterval(timer)
  }, [state, job, refresh])

  // The sample's file-pair gates arrive in the background (api.sample_gates).
  useEffect(() => {
    if (!pairs.pending) return undefined
    const timer = setInterval(async () => setPairs(await getJSON('/pairs')), 3000)
    return () => clearInterval(timer)
  }, [pairs.pending])

  async function runJob(path, init) {
    setQa(null)
    setWhy(false)
    let current = { order: [], stages: {}, warnings: [], log: null, files: [], ...init }
    setJob(current)
    try {
      await streamJob(path, (e) => {
        current = applyEvent(current, e)
        setJob(current)
      })
    } catch (err) {
      current = { ...current, error: err.message, done: true }
      setJob(current)
    }
    if (!current.error) {
      await refresh()
      setJob(null)
    }
  }

  async function addFiles(files) {
    if (!files.length) return
    setUploadError(null)
    let pending
    try {
      pending = (await uploadFiles(files)).pending
    } catch (err) {
      setUploadError(err.message)
      return
    }
    await runJob('/build', { files: pending, order: BUILD_STAGES })
  }

  async function startOver() {
    await sendJSON('/reset', 'POST')
    setJob(null)
    setQa(null)
    await refresh()
  }

  async function ask(question, withBaseline = compare) {
    setQa({ question, loading: true })
    setHover(null)
    setFocusN(null)
    setWhy(false)
    try {
      const data = await sendJSON('/ask', 'POST', { question, compare_baseline: withBaseline, lang })
      setQa({ question, data })
      setState((s) => ({ ...s, cost_usd: data.cost_usd, cost_known: data.cost_known }))
    } catch (err) {
      setQa({ question, error: err.message })
    }
  }
  function toggleCompare(on) {
    setCompare(on)
    if (qa?.question && !qa.loading) ask(qa.question, on)
  }

  async function setPair(pair, setting) {
    const body = await sendJSON('/pairs', 'PUT', { a: pair.a, b: pair.b, setting })
    setPairs(body)
    setStats(body.stats)
    setTimeline(await getJSON('/timeline'))     // the slide intervals come from audio_slide links
  }

  // One evidence list for the rail: ours, then anything only the baseline retrieved. An answer
  // that abstains rests on nothing, so its evidence is not shown.
  const items = useMemo(() => {
    const data = qa?.data
    if (!data) return []
    const mine = data.abstained ? [] : data.evidence
    const ours = new Set(mine.map((e) => e.unit_id))
    const theirs = (data.baseline && !data.baseline.abstained ? data.baseline.evidence : [])
      .filter((e) => !ours.has(e.unit_id)).map((e) => ({ ...e, baselineOnly: true }))
    return [...mine, ...theirs]
  }, [qa])
  const byN = useMemo(() => Object.fromEntries(items.map((e) => [e.n, e])), [items])
  const hoveredClaim = useMemo(() => {
    if (!hover || !qa?.data) return null
    const [side, i] = hover.split(':')
    return (side === 'baseline' ? qa.data.baseline : qa.data)?.answer_claims[Number(i)]
  }, [hover, qa])
  const highlight = new Set([...(hoveredClaim?.citations ?? []), ...(focusN != null ? [focusN] : [])])
  const missed = new Set(qa?.data?.baseline?.missed_evidence ?? [])
  const citedN = new Set((qa?.data?.answer_claims ?? []).flatMap((c) => c.citations))
  const ranges = items.filter((e) => e.modality === 'speech' && citedN.has(e.n)).map((e) => [e.location.start_s, e.location.end_s])

  function seek(seconds, { play = false, until = null } = {}) {
    setTime(seconds)
    stopAt.current = until
    const audio = audioRef.current
    if (!audio) return
    audio.currentTime = seconds
    if (play) audio.play().catch(() => {})
  }
  function toggle() {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) audio.play().catch(() => {})
    else audio.pause()
  }
  function playSegment(item) {
    const { start_s: start, end_s: end } = item.location
    if (playing && time >= start && time <= end) return audioRef.current?.pause()
    seek(start, { play: true, until: end })
  }
  /** A citation: light its card; a speech citation seeks the recording, a slide seeks to where
   * the slide was on screen (so the filmstrip marks it). */
  function cite(item) {
    setFocusN(item.n)
    if (item.modality === 'speech') seek(item.location.start_s)
    else if (item.modality === 'slides') {
      const shown = timeline?.intervals.find((iv) => iv.page === item.location.page && iv.file === item.file)
      if (shown) seek(shown.start)
    }
  }
  const audioProps = {
    onTimeUpdate: (e) => {
      const audio = e.currentTarget
      setTime(audio.currentTime)
      if (stopAt.current != null && audio.currentTime >= stopAt.current) {
        audio.pause()
        stopAt.current = null
      }
    },
    onPlay: () => setPlaying(true),
    onPause: () => setPlaying(false),
    onEnded: () => setPlaying(false),
  }
  const closeWhy = useCallback(() => {
    setWhy(false)
    requestAnimationFrame(() => document.getElementById('why-evidence')?.focus())
  }, [])

  async function home() {
    if (state?.corpus && !window.confirm(t.startOver)) return
    await startOver()
  }

  const corpus = state?.corpus
  const answerer = state?.answerer
  const ready = Boolean(corpus && !corpus.stale && !job && !state.building)
  const controls = ready && {
    compare,
    setCompare: toggleCompare,
    lang,
    setLang,
    cost: `${state.cost_known ? `$${state.cost_usd.toFixed(4)}` : t.priceUnknown}${answerer.mock ? ` · ${t.mock}` : ''}`,
    costTitle: `${answerer.model} · ${answerer.judge} · cap $${state.max_cost_usd}`,
  }

  let body = null
  if (job || state?.building) {
    body = <Building t={t} job={job ?? { order: [], stages: {}, warnings: [], files: state.pending ?? [], log: null }}
      onStartOver={startOver} />
  } else if (ready) {
    body = (
      <>
        <div className="flex min-h-0 flex-1">
          <Materials t={t} files={corpus.files} pairs={pairs.pairs} pending={pairs.pending} stats={stats}
            readOnly={corpus.sample} onSetPair={setPair} onAddFiles={addFiles} />
          <main className="flex min-w-0 flex-1 flex-col">
            <div className="min-h-0 flex-1 overflow-y-auto px-6">
              <div className={`mx-auto w-full ${qa?.data?.baseline ? 'max-w-[768px]' : 'max-w-[640px]'}`}>
                <AnswerPane t={t} qa={qa} hover={hover} setHover={setHover} onCite={cite} byN={byN}
                  onWhy={() => setWhy(true)} />
              </div>
            </div>
            <div className="px-6 pt-2 pb-5">
              <Composer t={t} busy={qa?.loading} onAsk={(q) => ask(q)} />
            </div>
          </main>
          <Evidence t={t} items={items} highlight={highlight} missed={missed} focusN={focusN}
            time={time} playing={playing} onPlay={playSegment} />
        </div>
        <Player t={t} timeline={timeline} audioRef={audioRef} time={time} playing={playing} ranges={ranges}
          onSeek={seek} onToggle={toggle} audioProps={audioProps} />
      </>
    )
  } else if (state) {
    body = <Empty t={t} sampleAvailable={state.sample_available} error={uploadError} onFiles={addFiles}
      onSample={() => runJob('/sample', { sample: true, order: ['index'] })} />
  }

  const sheet = why && ready && qa?.data
  return (
    <div className="flex h-full flex-col">
      {/* behind an open sheet nothing takes focus or clicks: the dialog is modal */}
      <div className="contents" inert={sheet ? true : undefined}>
        <TopBar t={t} title={ready ? corpus.title : null} onHome={home} controls={controls} />
        {body}
      </div>
      {sheet && <WhySheet t={t} evidence={qa.data.evidence} onClose={closeWhy} />}
    </div>
  )
}
