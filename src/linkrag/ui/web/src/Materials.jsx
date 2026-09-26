import { useRef } from 'react'
import { LINK_TYPES, clock } from './lib.js'
import { Dot } from './ui.jsx'

export const ACCEPT = '.mp3,.wav,.m4a,.pdf,.docx,.png,.jpg,.jpeg,.tif,.tiff'

function meta(file, t) {
  if (file.kind === 'speech') return clock(file.duration)
  if (file.pages) return file.kind === 'slides' ? t.meta.slides(file.pages) : t.meta.pages(file.pages)
  return t.meta.passages(file.units)
}

function FileRow({ file, t }) {
  return (
    <div className="flex h-8 items-center gap-2.5">
      <Dot modality={file.kind} />
      <span className="min-w-0 flex-1 truncate text-[14px] leading-5" title={file.name}>{file.name}</span>
      <span className="shrink-0 font-mono text-[12px] leading-5 text-muted">{meta(file, t)}</span>
    </div>
  )
}

/** The gate's verdict for a pair: solid = linked, dashed = unsure (with Confirm), faint = not
 * linked. Clicking it toggles related / unrelated. */
function PairControl({ pair, t, onSet, className = '' }) {
  const label = { linked: t.linked, unsure: t.unsure, unlinked: t.unlinked }[pair.state]
  return (
    <div className={`flex min-w-0 items-center gap-2 ${className}`}>
      <button type="button" onClick={() => onSet(pair, pair.links ? 'unrelated' : 'related')}
        title={t.pairTitle(pair)} aria-label={t.pairTitle(pair)}
        className="rounded text-[12px] leading-4 text-muted hover:text-ink">
        {label}
        {pair.z != null && <> · <span className="font-mono">{pair.z.toFixed(1)}</span></>}
      </button>
      {pair.state === 'unsure' && (
        <button type="button" onClick={() => onSet(pair, 'related')} title={t.confirmTitle}
          className="ml-auto h-6 rounded-md border border-line px-2 text-[12px] leading-none text-ink hover:bg-fill">
          {t.confirm}
        </button>
      )}
    </div>
  )
}

const LINE = { linked: 'border-ink', unsure: 'border-dashed border-ink', unlinked: 'border-dashed border-line' }

function Connector({ pair, t, onSet }) {
  return (
    <div className="relative flex min-h-7 items-center py-0.5 pl-[18px]">
      {/* from the dot above to the dot below: rows are 32 px, dots sit 12 px from their edges */}
      <span aria-hidden="true" className={`absolute -top-3 -bottom-3 left-[3.5px] border-l ${LINE[pair.state]}`} />
      <PairControl pair={pair} t={t} onSet={onSet} className="flex-1" />
    </div>
  )
}

export default function Materials({ t, files, pairs, stats, readOnly, onSetPair, onAddFiles }) {
  const input = useRef(null)
  const pairOf = (a, b) => pairs.find((p) => (p.a === a && p.b === b) || (p.a === b && p.b === a))
  const between = files.slice(1).map((file, i) => pairOf(files[i].name, file.name))
  const others = pairs.filter((p) => !between.includes(p))
  return (
    <aside aria-labelledby="materials-label" className="flex w-[264px] shrink-0 flex-col overflow-y-auto border-r border-hair px-5 py-6">
      <h2 id="materials-label" className="label">{t.materials}</h2>
      <ol className="mt-2">
        {files.map((file, i) => (
          <li key={file.name}>
            <FileRow file={file} t={t} />
            {between[i] ? <Connector pair={between[i]} t={t} onSet={onSetPair} /> : i + 1 < files.length && <div className="h-1" />}
          </li>
        ))}
      </ol>
      {others.length > 0 && (
        <div className="mt-4">
          <h3 className="label">{t.otherPairs}</h3>
          <ul className="mt-1">
            {others.map((p) => (
              <li key={`${p.a}|${p.b}`} className="py-1">
                <div className="truncate text-[12px] leading-4 text-ink" title={`${p.a} · ${p.b}`}>{p.a} · {p.b}</div>
                <PairControl pair={p} t={t} onSet={onSetPair} />
              </li>
            ))}
          </ul>
        </div>
      )}
      <input ref={input} type="file" multiple hidden accept={ACCEPT}
        onChange={(e) => { onAddFiles([...e.target.files]); e.target.value = '' }} />
      <button type="button" onClick={() => input.current.click()} disabled={readOnly}
        title={readOnly ? t.sampleReadOnly : undefined}
        className="mt-5 flex h-8 w-fit items-center gap-1.5 rounded-lg border border-line px-3 text-[13px] leading-none text-ink hover:bg-fill disabled:text-muted disabled:hover:bg-transparent">
        <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
          <path d="M6 1.5v9M1.5 6h9" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
        </svg>
        {t.addFile}
      </button>
      <hr className="my-6 border-hair" />
      <div className="flex items-baseline justify-between">
        <h2 className="label">{t.links}</h2>
        <span className="font-mono text-[12px] leading-4 text-muted">{stats?.total ?? 0}</span>
      </div>
      <dl className="mt-2">
        {LINK_TYPES.map((type) => (
          <div key={type} className="flex items-center justify-between text-[14px] leading-7">
            <dt>{t.linkRow[type]}</dt>
            <dd className="font-mono text-[13px] text-muted">{stats?.by_type?.[type] ?? 0}</dd>
          </div>
        ))}
      </dl>
    </aside>
  )
}
