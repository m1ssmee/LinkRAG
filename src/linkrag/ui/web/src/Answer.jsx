import { Fragment, useRef, useState } from 'react'
import { Chip, VerdictIcon } from './ui.jsx'

const VERDICTS = ['supported', 'weak', 'unsupported']

function summary(claims, t) {
  const parts = VERDICTS.map((v) => [v, claims.filter((c) => c.verdict === v).length])
    .filter(([, n]) => n)
    .map(([v, n]) => `${n} ${t.counts[v]}`)
  return t.summary(claims.length, parts.join(', '))
}

/** Claims as a list: hover or focus washes a claim and lights the evidence it cites;
 * ArrowUp / ArrowDown move between claims, Tab walks their citations. */
function Claims({ claims, side, t, byN, hover, setHover, onCite }) {
  const list = useRef(null)
  const step = (e, i) => {
    const d = { ArrowDown: 1, ArrowUp: -1 }[e.key]
    if (!d || e.target !== e.currentTarget) return
    e.preventDefault()
    list.current.querySelectorAll('[data-claim]')[i + d]?.focus()
  }
  const leave = (key) => setHover((h) => (h === key ? null : h))
  return (
    <ol ref={list} className="mt-2">
      {claims.map((c, i) => {
        const key = `${side}:${i}`
        return (
          <li key={key} data-claim tabIndex={0} onKeyDown={(e) => step(e, i)}
            onMouseEnter={() => setHover(key)} onMouseLeave={() => leave(key)} onFocus={() => setHover(key)}
            onBlur={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) leave(key) }}
            className={`-mx-3 flex gap-3 rounded-lg px-3 py-1.5 transition-colors ${hover === key ? 'bg-wash' : ''}`}>
            <VerdictIcon verdict={c.verdict} t={t} />
            <p className="min-w-0 text-[16px] leading-[26px]">
              {c.text}
              {c.citations.map((n) => byN[n] && (
                <Fragment key={n}>{' '}<Chip item={byN[n]} t={t} onClick={onCite} /></Fragment>
              ))}
            </p>
          </li>
        )
      })}
    </ol>
  )
}

/** One answer: header with its source counts, the claims or the abstention, the summary line. */
export function AnswerBlock({ label, answer, side, links, t, byN, hover, setHover, onCite, onWhy }) {
  const via = answer.evidence.filter((e) => e.origin === 'via_link').length
  return (
    <section aria-label={label}>
      <div className="flex flex-wrap items-baseline gap-x-1.5">
        <h2 className="label">{label}</h2>
        {!answer.abstained && (
          <span className="text-[12px] leading-4 text-muted">
            · <span className="font-mono">{answer.evidence.length}</span> {t.sources(answer.evidence.length)}
            {side === 'lectern' && (links ? <> · <span className="font-mono">{via}</span> {t.viaLinks}</> : <> · {t.noLinks}</>)}
          </span>
        )}
      </div>
      {answer.abstained ? (
        <p className="mt-3 text-[16px] leading-[26px] text-muted">{t.notFound}</p>
      ) : (
        <>
          <Claims claims={answer.answer_claims} side={side} t={t} byN={byN} hover={hover} setHover={setHover} onCite={onCite} />
          <p className="mt-3 text-[14px] leading-[22px] text-muted">
            {summary(answer.answer_claims, t)}
            {onWhy && (
              <>
                {' '}
                <button type="button" id="why-evidence" onClick={onWhy}
                  className="rounded text-ink underline decoration-line underline-offset-4 hover:decoration-ink">
                  {t.why}
                </button>
              </>
            )}
          </p>
          {answer.verification_note && <p className="mt-1 text-[12px] leading-4 text-muted">{answer.verification_note}</p>}
        </>
      )}
    </section>
  )
}

export function Composer({ t, busy, onAsk }) {
  const [text, setText] = useState('')
  const submit = (e) => {
    e.preventDefault()
    if (!text.trim() || busy) return
    onAsk(text.trim())
    setText('')
  }
  return (
    <form onSubmit={submit}
      className="mx-auto flex h-[52px] w-full max-w-[640px] items-center gap-2 rounded-[12px] border border-line bg-white pr-2 pl-4 focus-within:outline-2 focus-within:outline-offset-2 focus-within:outline-speech">
      <label htmlFor="question" className="sr-only">{t.question}</label>
      <input id="question" value={text} onChange={(e) => setText(e.target.value)} placeholder={t.askPlaceholder}
        autoComplete="off" className="h-full min-w-0 flex-1 bg-transparent text-[16px] outline-none placeholder:text-muted" />
      <button type="submit" disabled={busy || !text.trim()} aria-label={t.send}
        className="grid size-9 shrink-0 place-items-center rounded-[8px] bg-ink text-white disabled:bg-line">
        <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
          <path d="M7 12V2M2.5 6.5 7 2l4.5 4.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </form>
  )
}

/** The reading column: Question, then the answer (or the intro before the first question). */
export default function AnswerPane({ t, qa, hover, setHover, onCite, onWhy, byN }) {
  if (!qa) return <p className="pt-10 text-[16px] leading-[26px] text-muted">{t.intro}</p>
  const data = qa.data
  return (
    <div className="pt-10 pb-8">
      <section aria-labelledby="question-label">
        <h2 id="question-label" className="label">{t.question}</h2>
        <p className="mt-1.5 text-[22px] leading-[30px] font-medium">{qa.question}</p>
      </section>
      <div className="mt-8">
        {qa.loading && <p className="text-[14px] leading-[22px] text-muted" role="status">{t.working}</p>}
        {qa.error && <p className="text-[14px] leading-[22px] text-bad" role="alert">{qa.error}</p>}
        {data && !data.baseline && (
          <AnswerBlock label={t.answer} answer={data} side="lectern" links={data.links} t={t} byN={byN}
            hover={hover} setHover={setHover} onCite={onCite} onWhy={onWhy} />
        )}
        {data?.baseline && (
          /* State 4: the same question answered twice, the same claim styling side by side */
          <div className="grid grid-cols-2 gap-8">
            <AnswerBlock label={t.baseline} answer={data.baseline} side="baseline" t={t} byN={byN}
              hover={hover} setHover={setHover} onCite={onCite} />
            <AnswerBlock label={t.lectern} answer={data} side="lectern" links={data.links} t={t} byN={byN}
              hover={hover} setHover={setHover} onCite={onCite} onWhy={onWhy} />
          </div>
        )}
      </div>
    </div>
  )
}
