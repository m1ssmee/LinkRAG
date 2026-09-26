import { useEffect, useRef, useState } from 'react'
import { clock, getJSON } from './lib.js'
import { placeText } from './strings.js'
import { Chip, MOD } from './ui.jsx'

const COLUMNS = ['speech', 'slides', 'notes']
const WIDTH = 440
const ROW = 58
const TOP = 30

function short(unit, t) {
  const loc = unit.location || {}
  if (unit.modality === 'speech') return clock(loc.start_s)
  if (loc.page != null) return `${unit.modality === 'slides' ? t.slide : t.page} ${loc.page}${unit.figure ? ` · ${t.figure}` : ''}`
  return unit.file
}

/** State 6: the evidence as a sub-graph -- a node per unit (seeds that are not themselves
 * evidence are hollow), an edge per link between them, labelled by its type. */
export default function WhySheet({ t, evidence, onClose }) {
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState(null)
  const closeButton = useRef(null)
  useEffect(() => {
    const ids = [...new Set([...evidence.map((e) => e.unit_id), ...evidence.map((e) => e.via_unit).filter(Boolean)])]
    getJSON(`/graph?${new URLSearchParams(ids.map((id) => ['units', id]))}`).then(setGraph, (e) => setError(e.message))
    closeButton.current?.focus()
    const escape = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', escape)
    return () => window.removeEventListener('keydown', escape)
  }, [evidence, onClose])

  const nOf = Object.fromEntries(evidence.map((e) => [e.unit_id, e.n]))
  const direct = evidence.filter((e) => e.origin !== 'via_link').length
  let layout = null
  if (graph) {
    const columns = COLUMNS.filter((m) => graph.nodes.some((n) => n.modality === m))
    const at = {}
    let rows = 0
    columns.forEach((m, c) => {
      const nodes = graph.nodes.filter((n) => n.modality === m)
        .sort((a, b) => (nOf[a.unit_id] ?? 99) - (nOf[b.unit_id] ?? 99) || a.unit_id.localeCompare(b.unit_id))
      nodes.forEach((n, r) => { at[n.unit_id] = { x: 24 + (c * (WIDTH - 24)) / columns.length, y: TOP + r * ROW, n } })
      rows = Math.max(rows, nodes.length)
    })
    layout = { columns, at, height: TOP + rows * ROW }
  }
  const byId = graph ? Object.fromEntries(graph.nodes.map((n) => [n.unit_id, n])) : {}

  return (
    <div role="dialog" aria-modal="true" aria-labelledby="why-title"
      className="fixed top-14 right-0 bottom-[104px] z-20 flex w-[480px] flex-col border-l border-hair bg-white">
      <div className="flex h-12 shrink-0 items-center justify-between border-b border-hair px-5">
        <h2 id="why-title" className="text-[14px] leading-5 font-medium">{t.why}</h2>
        <button ref={closeButton} type="button" onClick={onClose} aria-label={t.close}
          className="grid size-7 place-items-center rounded-md text-muted hover:bg-fill hover:text-ink">
          <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
            <path d="M2 2l8 8M10 2l-8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        </button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
        <p className="text-[14px] leading-[22px] text-muted">{t.whyIntro(evidence.length, direct, evidence.length - direct)}</p>
        {error && <p role="alert" className="mt-3 text-[14px] text-bad">{error}</p>}
        {layout && (
          <>
            <div className="mt-4 flex gap-4 text-[12px] leading-4 text-muted" aria-hidden="true">
              {layout.columns.map((m) => (
                <span key={m} className="flex items-center gap-1.5">
                  <span className={`size-2 rounded-full ${MOD[m].dot}`} />{t.kinds[m]}
                </span>
              ))}
            </div>
            <svg width={WIDTH} height={layout.height} className="mt-2" role="img"
              aria-label={t.graphLabel(graph.nodes.length, graph.edges.length)}>
              {graph.edges.map((e) => {
                const a = layout.at[e.src]
                const b = layout.at[e.dst]
                if (!a || !b) return null
                const same = a.x === b.x
                const mx = same ? a.x + 44 : (a.x + b.x) / 2
                const my = (a.y + b.y) / 2
                const label = t.linkType[e.link_type] ?? e.link_type
                return (
                  <g key={`${e.src}>${e.dst}>${e.link_type}`}>
                    <path d={same ? `M${a.x} ${a.y} Q${a.x + 88} ${my} ${b.x} ${b.y}` : `M${a.x} ${a.y} L${b.x} ${b.y}`}
                      fill="none" stroke="#0A0A0A" strokeOpacity="0.35" strokeWidth="1" />
                    <rect x={mx - label.length * 2.9 - 4} y={my - 8} width={label.length * 5.8 + 8} height={16} rx={3} fill="#FFFFFF" />
                    <text x={mx} y={my + 3.5} textAnchor="middle" fontSize="10" fill="#6B6B6B">{label}</text>
                  </g>
                )
              })}
              {Object.values(layout.at).map(({ x, y, n }) => {
                const inEvidence = nOf[n.unit_id] != null
                return (
                  <g key={n.unit_id}>
                    <circle cx={x} cy={y} r={6} fill={inEvidence ? MOD[n.modality].stroke : '#FFFFFF'}
                      stroke={MOD[n.modality].stroke} strokeWidth="1.5" />
                    <text x={x + 12} y={y + 4} fontSize="12" fill="#0A0A0A">
                      {inEvidence && <tspan fontFamily="Geist Mono, monospace" fontWeight="500">{nOf[n.unit_id]} </tspan>}
                      <tspan fill={inEvidence ? '#0A0A0A' : '#6B6B6B'}>{short(n, t)}</tspan>
                    </text>
                  </g>
                )
              })}
            </svg>
            <h3 className="label mt-6">{t.linksBetween}</h3>
            {graph.edges.length === 0 ? (
              <p className="mt-2 text-[14px] leading-[22px] text-muted">{t.noEdges}</p>
            ) : (
              <ul className="mt-2 space-y-1.5">
                {graph.edges.map((e) => {
                  const ends = [e.src, e.dst].map((id) => ({ ...byId[id], n: nOf[id] }))
                  return (
                    <li key={`${e.src}>${e.dst}>${e.link_type}`} className="flex flex-wrap items-center gap-1.5 text-[13px] leading-5">
                      {ends.map((u, i) => (
                        <span key={u.unit_id} className="flex items-center gap-1.5">
                          {i === 1 && <span aria-hidden="true" className="text-muted">→</span>}
                          {u.n != null ? <Chip item={u} t={t} /> : <span className="text-muted">{t.seedOnly}</span>}
                          <span>{placeText(u, t, clock)}</span>
                        </span>
                      ))}
                      <span className="text-muted">· {t.linkType[e.link_type] ?? e.link_type} · <span className="font-mono">{e.score.toFixed(2)}</span></span>
                    </li>
                  )
                })}
              </ul>
            )}
          </>
        )}
      </div>
    </div>
  )
}
