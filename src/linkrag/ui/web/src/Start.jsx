import { useEffect, useRef, useState } from 'react'
import { ACCEPT } from './Materials.jsx'
import { clock } from './lib.js'
import { Dot } from './ui.jsx'

const KINDS = [
  { kind: 'speech', ext: '.mp3 .wav .m4a' },
  { kind: 'slides', ext: '.pdf' },
  { kind: 'notes', ext: '.pdf .docx .png .jpg' },
]

/** State 1: a drop zone naming the three kinds of material, and the sample. Nothing else. */
export function Empty({ t, sampleAvailable, onFiles, onSample, error }) {
  const [over, setOver] = useState(false)
  const input = useRef(null)
  const choose = () => input.current.click()
  return (
    <main className="grid flex-1 place-items-center px-6">
      <div className="w-full max-w-[560px]">
        <div role="button" tabIndex={0} aria-label={t.dropLabel} onClick={choose}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose() } }}
          onDragOver={(e) => { e.preventDefault(); setOver(true) }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); onFiles([...e.dataTransfer.files]) }}
          className={`rounded-2xl border border-dashed px-10 py-12 text-center transition-colors ${over ? 'border-speech bg-wash' : 'border-line'}`}>
          <p className="text-[16px] leading-[26px] font-medium">{t.dropTitle}</p>
          <p className="text-[14px] leading-[22px] text-muted">{t.dropHint}</p>
          <ul className="mx-auto mt-8 grid w-fit grid-cols-[auto_auto_auto] items-center gap-x-3 gap-y-2 text-left">
            {KINDS.map(({ kind, ext }) => (
              <li key={kind} className="contents">
                <Dot modality={kind} />
                <span className="text-[14px] leading-[22px]">
                  {t.kinds[kind]} <span className="text-muted">· {t.kindHints[kind]}</span>
                </span>
                <span className="font-mono text-[12px] leading-[22px] text-muted">{ext}</span>
              </li>
            ))}
          </ul>
        </div>
        <input ref={input} type="file" multiple hidden accept={ACCEPT}
          onChange={(e) => { onFiles([...e.target.files]); e.target.value = '' }} />
        {error && <p role="alert" className="mt-3 text-[14px] leading-[22px] text-bad">{error}</p>}
        {sampleAvailable && (
          <div className="mt-5 text-center">
            <button type="button" onClick={onSample}
              className="h-8 rounded-lg border border-line px-3 text-[13px] leading-none text-ink hover:bg-fill">
              {t.trySample}
            </button>
          </div>
        )}
      </div>
    </main>
  )
}

function useNow(active) {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    if (!active) return undefined
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [active])
  return now
}

function StageIcon({ status }) {
  if (status === 'done') {
    return (
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M3.5 8.5l3 3 6-7" fill="none" stroke="#0A0A0A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  }
  if (status === 'running') {
    return (
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true" className="animate-spin">
        <circle cx="8" cy="8" r="6" fill="none" stroke="#E5E5E5" strokeWidth="1.5" />
        <path d="M8 2a6 6 0 0 1 6 6" fill="none" stroke="#2F5BD3" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    )
  }
  if (status === 'error') {
    return (
      <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
        <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" fill="none" stroke="#C62828" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    )
  }
  return (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="5.5" fill="none" stroke="#E5E5E5" strokeWidth="1.5" />
    </svg>
  )
}

/** State 2: the pipeline's stages as they run, each with its time and latest log line. */
export function Building({ t, job, onStartOver }) {
  const now = useNow(!job.done)
  const failed = job.error != null
  const hasSpeech = job.files.some((name) => /\.(mp3|wav|m4a)$/i.test(name))
  return (
    <main className="grid flex-1 place-items-center px-6">
      <div className="w-full max-w-[480px]" aria-live="polite">
        <h1 className="label">{job.sample ? t.loadingSample : t.building}</h1>
        {/* no modality dots yet: whether a PDF is a deck or notes is decided at ingest */}
        {job.files.length > 0 && (
          <ul className="mt-3 space-y-0.5">
            {job.files.map((name) => (
              <li key={name} className="truncate text-[14px] leading-6">{name}</li>
            ))}
          </ul>
        )}
        <ol className="mt-6 space-y-3">
          {job.order.map((name) => {
            const stage = job.stages[name] ?? {}
            const status = failed && stage.status === 'running' ? 'error' : stage.status ?? 'pending'
            const seconds = stage.seconds ?? (stage.started ? (now - stage.started) / 1000 : null)
            return (
              <li key={name}>
                <div className="flex items-center gap-3">
                  <StageIcon status={status} />
                  <span className={`text-[14px] leading-[22px] ${status === 'pending' ? 'text-muted' : 'text-ink'}`}>
                    {t.stages[name] ?? name}
                  </span>
                  {seconds != null && (
                    <span className="ml-auto font-mono text-[12px] leading-[22px] text-muted">{clock(seconds)}</span>
                  )}
                </div>
                {status === 'running' && name === 'ingest' && hasSpeech && (
                  <p className="mt-0.5 pl-7 text-[12px] leading-4 text-muted">{t.transcribing}</p>
                )}
                {status === 'running' && job.log && (
                  <p className="mt-0.5 truncate pl-7 font-mono text-[12px] leading-4 text-muted" title={job.log}>{job.log}</p>
                )}
              </li>
            )
          })}
        </ol>
        {job.queued && !job.done && <p className="mt-4 text-[12px] leading-4 text-muted">{t.queued}</p>}
        {job.warnings.map((w) => (
          <p key={w} className="mt-3 text-[12px] leading-4 text-muted">{w}</p>
        ))}
        {failed && (
          <div className="mt-5">
            <p role="alert" className="text-[14px] leading-[22px] text-bad">{job.error}</p>
            <button type="button" onClick={onStartOver}
              className="mt-3 h-8 rounded-lg border border-line px-3 text-[13px] leading-none text-ink hover:bg-fill">
              {t.startAgain}
            </button>
          </div>
        )}
      </div>
    </main>
  )
}
