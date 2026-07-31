// ═══════════════════════════════════════════════════════════════════════
// Engine read surfaces — shared helpers
// ═══════════════════════════════════════════════════════════════════════
// The headless portfolio engines all answer with the same contract shape, so
// they share one fetch path and one empty-state vocabulary. 4.10a
// reconciliation was the first consumer; 3.5b event radar and 5.5 fund flows
// followed, and this file exists because those two now live on a different
// page (/monitor) from the reconciliation panel (/portfolio).
//
// It centralises exactly one rule, which is the rule these engines exist to
// protect: a panel with nothing to show must render the ENGINE's reason for
// having nothing to say. A blank success state is a different claim from an
// absent recorder, and only the engine knows which one is true — so the
// renderer is never called with a fault, and never has to guess.
//
// Loaded via <script src>, which is also the point. Each panel's renderer sits
// in its own page's inline block, so a parse error in one no longer takes the
// others — or the holdings grid's Save button — down with it.

function esc(s) {
  return String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function engineNotice(tone, headline, detail) {
  const tones = {
    error:   { icon: 'error',         cls: 'text-error',     ring: 'border-error/20 bg-error/5' },
    waiting: { icon: 'hourglass_top', cls: 'text-secondary', ring: 'border-secondary/20 bg-secondary/5' },
    warn:    { icon: 'warning',       cls: 'text-[#e0b252]', ring: 'border-[#e0b252]/20 bg-[#e0b252]/5' },
    ok:      { icon: 'check_circle',  cls: 'text-primary',   ring: 'border-primary/20 bg-primary/5' }
  };
  const t = tones[tone] || tones.waiting;
  return `<div class="flex items-start gap-3 px-5 py-4 rounded-xl border ${t.ring}">
      <span class="material-symbols-outlined text-base ${t.cls} shrink-0" data-icon="${t.icon}">${t.icon}</span>
      <div>
        <p class="text-[11px] font-black uppercase tracking-[0.15em] ${t.cls}">${esc(headline)}</p>
        <p class="text-[10px] text-on-surface-variant/60 mt-1.5 leading-relaxed max-w-3xl">${esc(detail)}</p>
      </div>
    </div>`;
}

async function loadEnginePanel(url, bodyId, render) {
  const body = document.getElementById(bodyId);
  if (!body) return;
  try {
    const res = await fetch(url);
    if (!res.ok) {
      body.innerHTML = engineNotice('error', 'Engine unavailable',
        `The read path returned HTTP ${res.status}. This is a fault in the surface, not a statement about your portfolio.`);
      return;
    }
    render(await res.json(), body);
  } catch (err) {
    body.innerHTML = engineNotice('error', 'Engine unreachable',
      `${err}. This is a fault in the surface, not a statement about your portfolio.`);
  }
}

// Shared because reconciliation and fund flows both print a raw share count.
// An unmeasurable one is an em dash, never a 0 — the same rule both panels are
// built around, and it has to survive being read from two pages.
function fmtShares(v) {
  const n = Number(v);
  if (v === null || v === undefined || !isFinite(n)) return '—';
  return n.toLocaleString(undefined, { maximumFractionDigits: 4 });
}
