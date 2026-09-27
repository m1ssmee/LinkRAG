import { clock } from './lib.js'
import { placeText } from './strings.js'

export const MOD = {
  speech: { dot: 'bg-speech', chip: 'bg-speech-bg text-speech', stroke: '#2F5BD3' },
  slides: { dot: 'bg-slides', chip: 'bg-slides-bg text-slides-ink', stroke: '#B45309' },
  notes: { dot: 'bg-notes', chip: 'bg-notes-bg text-notes', stroke: '#404040' },
}

export function LogoMark({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" aria-hidden="true">
      <path d="M4.25 10.5A4.75 4.75 0 0 1 13.75 10.5" fill="none" stroke="#0A0A0A" strokeWidth="1.25" />
      <circle cx="4.25" cy="10.5" r="2.5" fill="#2F5BD3" />
      <circle cx="13.75" cy="10.5" r="2.5" fill="#B45309" />
    </svg>
  )
}

export function Logo() {
  return (
    <span className="flex items-center gap-2">
      <LogoMark />
      <span className="font-serif text-[22px] leading-none tracking-[-0.2px]">Lectern</span>
    </span>
  )
}

export const Dot = ({ modality, className = '' }) => (
  <span aria-hidden="true" className={`inline-block size-2 shrink-0 rounded-full ${MOD[modality].dot} ${className}`} />
)

/** A numbered citation chip; a button when it can be clicked. */
export function Chip({ item, t, onClick }) {
  const cls = `inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-[5px] px-1 font-mono text-[11px] font-medium leading-none ${MOD[item.modality].chip}`
  if (!onClick) return <span className={cls}>{item.n}</span>
  return (
    <button type="button" className={`${cls} align-[2px]`} onClick={() => onClick(item)}
      aria-label={t.chip(item.n, placeText(item, t, clock))}>
      {item.n}
    </button>
  )
}

/** "Audio · 18:40–19:05", "Slide 12 · figure", "Notes · page 4"; numbers in mono. */
export function Place({ unit, t }) {
  const loc = unit.location || {}
  const mono = (text) => <span className="font-mono">{text}</span>
  if (unit.modality === 'speech') return <>{t.audio} · {mono(`${clock(loc.start_s)}–${clock(loc.end_s)}`)}</>
  if (unit.modality === 'slides') return <>{t.slide} {mono(loc.page)}{unit.figure && ` · ${t.figure}`}</>
  if (loc.page != null) return <>{t.notes} · {t.page} {mono(loc.page)}</>
  return <>{t.notes} · {unit.figure ? t.image : unit.file}</>
}

/** Outline check = supported, dashed circle with "!" = weak, red outline x = unsupported. */
export function VerdictIcon({ verdict, t }) {
  const label = t.verdict[verdict] ?? t.verdict.unchecked
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" role="img" aria-label={label} className="mt-[5px] shrink-0">
      <title>{label}</title>
      {verdict === 'supported' && (
        <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="#0A0A0A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      )}
      {verdict === 'weak' && (
        <g fill="none" stroke="#6B6B6B" strokeWidth="1.25" strokeLinecap="round">
          <circle cx="8" cy="8" r="6.25" strokeDasharray="2.2 2.2" />
          <path d="M8 4.9v3.8" />
          <circle cx="8" cy="11.2" r="0.4" fill="#6B6B6B" />
        </g>
      )}
      {verdict === 'unsupported' && (
        <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="#C62828" strokeWidth="1.5" strokeLinecap="round" />
      )}
      {!['supported', 'weak', 'unsupported'].includes(verdict) && <circle cx="8" cy="8" r="2" fill="#6B6B6B" />}
    </svg>
  )
}

export const PlayIcon = ({ playing, size = 12 }) => (
  <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" fill="currentColor">
    {playing ? <path d="M2.5 1.5h2.5v9H2.5zM7 1.5h2.5v9H7z" /> : <path d="M3 1.5l7.5 4.5L3 10.5z" />}
  </svg>
)
