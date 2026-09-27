import { useEffect, useRef, useState } from 'react'
import { clock, withSession } from './lib.js'
import { placeText } from './strings.js'
import { Chip, Dot, Place, PlayIcon } from './ui.jsx'

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

function Card({ item, t, highlight, unretrieved, time, playing, onPlay }) {
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
      {unretrieved && (
        <span className="mt-2 inline-block rounded-[4px] border border-line px-1.5 text-[11px] leading-[18px] text-muted">{t.notRetrieved}</span>
      )}
      {/* dimmed to 40 % when the baseline did not retrieve it; full contrast again on hover or focus */}
      <div className={unretrieved ? 'opacity-40 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100' : ''}>
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

/** Context, not citations: units linked to the cited evidence that no claim cites. No number,
 * no verdict; the answer does not rest on them. */
function Linked({ t, rows, byN }) {
  const [open, setOpen] = useState(true)
  return (
    <section className="mt-5">
      <h3>
        <button type="button" aria-expanded={open} aria-controls="linked-rows" onClick={() => setOpen(!open)}
          className="label flex items-center gap-1.5 rounded hover:text-ink">
          {t.linkedTitle}
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true"
            className={`transition-transform ${open ? '' : '-rotate-90'}`}>
            <path d="M2 3.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </h3>
      {open && (
        <ul id="linked-rows" className="mt-1.5">
          {rows.map((row) => {
            const from = byN[row.from_n] ? t.linkedFrom(placeText(byN[row.from_n], t, clock)) : ''
            return (
              <li key={row.unit_id} title={from} className="flex h-7 items-center gap-2.5">
                <Dot modality={row.modality} />
                <span className="min-w-0 truncate text-[13px] leading-5 text-ink"><Place unit={row} t={t} /></span>
                <span className="ml-auto shrink-0 text-[12px] leading-5 text-muted">
                  {t.via(t.linkType[row.link_type] ?? row.link_type)}
                </span>
                <span className="sr-only">, {from}</span>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}

export default function Evidence({ t, items, linked, highlight, unretrieved, focusN, time, playing, onPlay }) {
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
            <Card key={item.n} item={item} t={t} highlight={highlight.has(item.n)} unretrieved={unretrieved.has(item.unit_id)}
              time={time} playing={playing} onPlay={onPlay} />
          ))}
        </ol>
      )}
      {items.length > 0 && linked.length > 0 && (
        <Linked t={t} rows={linked} byN={Object.fromEntries(items.map((e) => [e.n, e]))} />
      )}
    </aside>
  )
}
