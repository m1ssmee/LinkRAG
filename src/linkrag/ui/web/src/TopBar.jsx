import { Logo } from './ui.jsx'

function Switch({ checked, onChange, label }) {
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

function Segmented({ value, options, onChange, label }) {
  return (
    <div role="radiogroup" aria-label={label} className="flex h-7 items-center rounded-lg border border-line p-0.5">
      {options.map((o) => (
        <button key={o.value} type="button" role="radio" aria-checked={value === o.value} lang={o.lang}
          onClick={() => onChange(o.value)}
          className={`h-full rounded-md px-2 text-[12px] leading-none ${value === o.value ? 'bg-fill font-medium text-ink' : 'text-muted'}`}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

/** 56 px: logo, divider, lecture title; compare switch, language, session cost. */
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
          <Segmented label={t.language} value={controls.lang} onChange={controls.setLang}
            options={[{ value: 'en', label: 'EN', lang: 'en' }, { value: 'hi', label: 'हिंदी', lang: 'hi' }]} />
          <span className="font-mono text-[13px] leading-5 text-muted" title={controls.costTitle}>
            <span className="sr-only">{t.cost}: </span>
            {controls.cost}
          </span>
        </div>
      )}
    </header>
  )
}
