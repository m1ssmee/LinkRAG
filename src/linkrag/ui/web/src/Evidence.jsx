import { useEffect, useRef, useState } from 'react'
import { clock, withSession } from './lib.js'
import { placeText } from './strings.js'
import { Chip, Place, PlayIcon } from './ui.jsx'

function AudioRow({ item, t, time, playing, onPlay }) {
  const { start_s: start, end_s: end } = item.location
  const inside = time >= start && time <= end
  const range = `${clock(start)}–${clock(end)}`
  return (
    <div className="mt-2.5 flex items-center gap-2.5">
      <button type="button" onClick={() => onPlay(item)} aria-label={t.playSegment(range)}
        className="grid size-7 shrink-0 place-items-center rounded-full border border-line text-ink hover:bg-fill">
        <PlayIcon playing={inside && playing} size={10} />
      </button>
      <div aria-hidden="true" className="relative h-1 flex-1 overflow-hidden rounded-full bg-fill">
        <div className="absolute inset-y-0 left-0 bg-speech" style={{ width: `${inside ? ((time - start) / (end - start)) * 100 : 0}%` }} />
      </div>
      <span className="font-mono text-[12px] leading-4 text-muted">{clock(end - start)}</span>
    </div>
  )
}

function Card({ item, t, highlight, missed, time, playing, onPlay }) {
  const [broken, setBroken] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const origin = item.baselineOnly ? t.baselineOnly
    : item.origin === 'via_link' ? t.via(t.linkType[item.link_type] ?? item.link_type) : t.directMatch
  return (
    <li id={`evidence-${item.n}`}
      className={`group rounded-[10px] border bg-white p-3 transition-colors ${highlight ? 'border-cited' : 'border-hair'}`}>
      <div className="flex items-center gap-2">
        <Chip item={item} t={t} />
        <span className="min-w-0 truncate text-[13px] font-medium leading-5 text-ink"><Place unit={item} t={t} /></span>
        <span className="ml-auto shrink-0 text-[12px] leading-5 text-muted">{origin}</span>
      </div>
      {missed && (
        <span className="mt-2 inline-block rounded-[4px] border border-line px-1.5 text-[11px] leading-[18px] text-muted">{t.missed}</span>
      )}
      {/* dimmed to 40 % when the baseline missed it; full contrast again on hover or focus */}
      <div className={missed ? 'opacity-40 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100' : ''}>
        <p className="mt-2 text-[14px] leading-[22px] text-ink">{item.excerpt}</p>
        {item.crop_url && !broken && (
          <a href={withSession(item.crop_url)} target="_blank" rel="noreferrer" className="mt-2.5 block w-fit rounded-md">
            {/* hidden until loaded: a page is rendered on first request */}
            <img src={withSession(item.crop_url)} alt={placeText(item, t, clock)}
              onLoad={() => setLoaded(true)} onError={() => setBroken(true)}
              className={loaded ? 'max-h-[240px] max-w-full rounded-md border border-hair' : 'h-0'} />
          </a>
        )}
        {item.modality === 'speech' && item.location?.start_s != null && (
          <AudioRow item={item} t={t} time={time} playing={playing} onPlay={onPlay} />
        )}
      </div>
    </li>
  )
}

export default function Evidence({ t, items, highlight, missed, focusN, time, playing, onPlay }) {
  const rail = useRef(null)
  useEffect(() => {
    if (focusN != null) rail.current?.querySelector(`#evidence-${focusN}`)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' })
  }, [focusN])
  return (
    <aside ref={rail} aria-labelledby="evidence-label" className="w-[360px] shrink-0 overflow-y-auto border-l border-hair px-5 py-6">
      <h2 id="evidence-label" className="label">{t.evidence}</h2>
      {items.length === 0 ? (
        <p className="mt-3 text-[14px] leading-[22px] text-muted">{t.evidenceEmpty}</p>
      ) : (
        <ol className="mt-3 space-y-3">
          {items.map((item) => (
            <Card key={item.n} item={item} t={t} highlight={highlight.has(item.n)} missed={missed.has(item.unit_id)}
              time={time} playing={playing} onPlay={onPlay} />
          ))}
        </ol>
      )}
    </aside>
  )
}
