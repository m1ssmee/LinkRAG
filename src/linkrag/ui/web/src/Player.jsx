import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { clock, withSession } from './lib.js'
import { PlayIcon } from './ui.jsx'

const BAR = 4
const GAP = 2
const WAVE = 40

/** The server's peaks, one per bar that fits. */
function resample(peaks, bars) {
  if (bars <= 0) return []
  if (!peaks.length) return new Array(bars).fill(0)
  return Array.from({ length: bars }, (_, i) => {
    const a = Math.floor((i * peaks.length) / bars)
    const b = Math.max(a + 1, Math.floor(((i + 1) * peaks.length) / bars))
    return Math.max(...peaks.slice(a, b))
  })
}

/** 104 px: play, the slide filmstrip (active slide amber, numbered) over the waveform (cited
 * ranges blue), a blue playhead, and mm:ss / total. The track is a slider for the keyboard. */
export default function Player({ t, timeline, audioRef, time, playing, ranges, onSeek, onToggle, audioProps }) {
  const track = useRef(null)
  const [width, setWidth] = useState(0)
  useLayoutEffect(() => {
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(track.current)
    return () => observer.disconnect()
  }, [])
  const duration = timeline?.duration || 0
  const bars = Math.floor((width + GAP) / (BAR + GAP))
  const heights = useMemo(() => resample(timeline?.peaks || [], bars), [timeline, bars])
  const at = (s) => (duration ? Math.min(1, Math.max(0, s / duration)) : 0)
  const active = timeline?.intervals.find((iv) => time >= iv.start && time < iv.end)
  const seekTo = (e) => {
    const box = track.current.getBoundingClientRect()
    onSeek(((e.clientX - box.left) / box.width) * duration)
  }
  const keys = (e) => {
    const step = { ArrowLeft: -5, ArrowRight: 5, PageDown: -30, PageUp: 30 }[e.key]
    const to = step != null ? time + step : e.key === 'Home' ? 0 : e.key === 'End' ? duration : null
    if (to != null) {
      e.preventDefault()
      onSeek(Math.min(duration, Math.max(0, to)))
    }
  }
  return (
    <footer className="flex h-[104px] shrink-0 items-center gap-5 border-t border-hair px-5">
      <button type="button" onClick={onToggle} disabled={!timeline?.audio_url}
        aria-label={playing ? t.pauseRecording : t.playRecording}
        className="grid size-10 shrink-0 place-items-center rounded-full bg-ink text-white disabled:bg-line">
        <PlayIcon playing={playing} size={14} />
      </button>
      <div ref={track} role="slider" tabIndex={duration ? 0 : -1} aria-label={t.timeline}
        aria-valuemin={0} aria-valuemax={Math.round(duration)} aria-valuenow={Math.round(time)}
        aria-valuetext={`${clock(time)} / ${clock(duration)}`}
        onPointerDown={(e) => { if (!duration) return; e.currentTarget.setPointerCapture(e.pointerId); seekTo(e) }}
        onPointerMove={(e) => { if (duration && e.buttons === 1) seekTo(e) }}
        onKeyDown={keys}
        className="relative h-16 min-w-0 flex-1 touch-none select-none rounded-sm">
        <div aria-hidden="true" className="absolute inset-x-0 top-0 h-4">
          {timeline?.intervals.map((iv) => (
            <div key={`${iv.start}-${iv.unit_id}`} title={`${iv.file} p.${iv.page}`}
              className={`absolute top-0 h-4 rounded-[3px] ${iv === active ? 'z-10 flex min-w-[18px] items-center justify-center bg-slides' : 'bg-fill'}`}
              style={{ left: `${at(iv.start) * 100}%`, width: `calc(${(at(iv.end) - at(iv.start)) * 100}% - 1px)` }}>
              {iv === active && <span className="font-mono text-[10px] font-medium leading-none text-white">{iv.page}</span>}
            </div>
          ))}
        </div>
        <svg aria-hidden="true" className="absolute inset-x-0 bottom-0" width={width} height={WAVE}>
          {heights.map((h, i) => {
            const x = i * (BAR + GAP)
            const s = ((x + BAR / 2) / (width || 1)) * duration
            const cited = ranges.some(([a, b]) => s >= a && s <= b)
            const height = Math.max(2, h * WAVE)
            return <rect key={i} x={x} y={(WAVE - height) / 2} width={BAR} height={height} rx={1}
              fill={cited ? '#2F5BD3' : '#C4C4C4'} />
          })}
        </svg>
        <span aria-hidden="true" className="pointer-events-none absolute -top-1 -bottom-1 w-0.5 -translate-x-1/2 rounded-full bg-speech"
          style={{ left: `${at(time) * 100}%` }} />
      </div>
      <span className="shrink-0 font-mono text-[13px] leading-5 text-muted tabular-nums">
        {clock(time)} / {clock(duration)}
      </span>
      {timeline?.audio_url && <audio ref={audioRef} src={withSession(timeline.audio_url)} preload="metadata" {...audioProps} />}
    </footer>
  )
}
