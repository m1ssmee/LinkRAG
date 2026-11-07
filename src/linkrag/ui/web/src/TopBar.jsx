import { Logo } from './ui.jsx'

export function Switch({ checked, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}
      className="flex items-center gap-2 rounded-md text-[13px] leading-5 text-ink">
      <span aria-hidden="true"
        className={`relative h-4 w-7 rounded-full transition-colors ${checked ? 'bg-ink' : 'bg-line'}`}>
        <span className={`absolute top-0.5 left-0 size-3 rounded-full bg-white transition-transform ${checked ? 'translate-x-3.5' : 'translate-x-0.5'}`} />
      </span>
      {label}
    </button>
  )
}

/** 56 px: logo, divider, lecture title; compare switch, session cost. */
export default function TopBar({ t, title, onHome, controls }) {
  return (
    <header className="flex h-14 shrink-0 items-center border-b border-hair px-5">
      <button type="button" onClick={onHome} aria-label={t.home} className="rounded-md">
        <Logo />
      </button>
      {title && (
        <>
          <span aria-hidden="true" className="mx-4 h-5 w-px bg-hair" />
          <span className="min-w-0 truncate text-[14px] leading-5 text-muted">{title}</span>
        </>
      )}
      {controls && (
        <div className="ml-auto flex shrink-0 items-center gap-5 pl-6">
          <Switch checked={controls.compare} onChange={controls.setCompare} label={t.compare} />
          <span className="font-mono text-[13px] leading-5 text-muted" title={controls.costTitle}>
            <span className="sr-only">{t.cost}: </span>
            {controls.cost}
          </span>
        </div>
      )}
    </header>
  )
}
