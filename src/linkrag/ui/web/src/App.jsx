import { useEffect, useMemo, useRef, useState } from 'react'
import AnswerPane, { Composer } from './Answer.jsx'
import Evidence from './Evidence.jsx'
import Materials from './Materials.jsx'
import Player from './Player.jsx'
import TopBar from './TopBar.jsx'
import { getJSON, sendJSON, streamJob } from './lib.js'
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

export default function App() {
  const [lang, setLang] = useStored('lectern-lang', 'en')
  const [compare, setCompare] = useStored('lectern-compare', false)
  const t = STRINGS[lang]
  const [state, setState] = useState(null)
  const [pairs, setPairs] = useState([])
  const [stats, setStats] = useState(null)
  const [timeline, setTimeline] = useState(null)
  const [qa, setQa] = useState(null)
  const [hover, setHover] = useState(null)
  const [focusN, setFocusN] = useState(null)
  const [time, setTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef(null)
  const stopAt = useRef(null)

  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])

  async function refresh() {
    const next = await getJSON('/state')
    setState(next)
    if (next.corpus) {
      const [p, s, tl] = await Promise.all([getJSON('/pairs'), getJSON('/stats'), getJSON('/timeline')])
      setPairs(p.pairs)
      setStats(s)
      setTimeline(tl)
    }
    return next
  }
  useEffect(() => {
    refresh()
  }, [])

  async function ask(question) {
    setQa({ question, loading: true })
    setHover(null)
    setFocusN(null)
    try {
      // the two-column compare view lands with the states; until then no baseline is asked for
      const data = await sendJSON('/ask', 'POST', { question, compare_baseline: false, lang })
      setQa({ question, data })
      setState((s) => ({ ...s, cost_usd: data.cost_usd, cost_known: data.cost_known }))
    } catch (err) {
      setQa({ question, error: err.message })
    }
  }

  async function setPair(pair, setting) {
    const body = await sendJSON('/pairs', 'PUT', { a: pair.a, b: pair.b, setting })
    setPairs(body.pairs)
    setStats(body.stats)
    setTimeline(await getJSON('/timeline'))     // the slide intervals come from audio_slide links
  }

  // One evidence list for the rail: ours, then anything only the baseline retrieved.
  const items = useMemo(() => {
    const data = qa?.data
    if (!data) return []
    const ours = new Set(data.evidence.map((e) => e.unit_id))
    const theirs = (data.baseline?.evidence ?? []).filter((e) => !ours.has(e.unit_id)).map((e) => ({ ...e, baselineOnly: true }))
    return [...data.evidence, ...theirs]
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

  async function home() {
    if (state?.corpus && !window.confirm(t.startOver)) return
    await sendJSON('/reset', 'POST')
    setQa(null)
    await refresh()
  }

  const corpus = state?.corpus
  const answerer = state?.answerer
  const controls = corpus && {
    compare,
    setCompare,
    lang,
    setLang,
    cost: `${state.cost_known ? `$${state.cost_usd.toFixed(4)}` : t.priceUnknown}${answerer.mock ? ` · ${t.mock}` : ''}`,
    costTitle: `${answerer.model} · ${answerer.judge} · cap $${state.max_cost_usd}`,
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar t={t} title={corpus?.title} onHome={home} controls={controls} />
      {corpus ? (
        <>
          <div className="flex min-h-0 flex-1">
            <Materials t={t} files={corpus.files} pairs={pairs} stats={stats} readOnly={corpus.sample}
              onSetPair={setPair} onAddFiles={() => {}} />
            <main className="flex min-w-0 flex-1 flex-col">
              <div className="min-h-0 flex-1 overflow-y-auto px-6">
                <div className="mx-auto w-full max-w-[640px]">
                  <AnswerPane t={t} qa={qa} hover={hover} setHover={setHover} onCite={cite} byN={byN} />
                </div>
              </div>
              <div className="px-6 pt-2 pb-5">
                <Composer t={t} busy={qa?.loading} onAsk={ask} />
              </div>
            </main>
            <Evidence t={t} items={items} highlight={highlight} missed={missed} focusN={focusN}
              time={time} playing={playing} onPlay={playSegment} />
          </div>
          <Player t={t} timeline={timeline} audioRef={audioRef} time={time} playing={playing} ranges={ranges}
            onSeek={seek} onToggle={toggle} audioProps={audioProps} />
        </>
      ) : (
        state?.sample_available && (
          <main className="grid flex-1 place-items-center">
            <button type="button" className="rounded-lg border border-line px-4 py-2 text-[14px]"
              onClick={async () => { await streamJob('/sample', () => {}); await refresh() }}>
              Load the sample lecture
            </button>
          </main>
        )
      )}
    </div>
  )
}
