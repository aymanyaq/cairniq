// Global error logging for troubleshooting
window.addEventListener('error', (event) => {
    fetch('/api/logs/frontend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            level: 'error',
            phase: 'UI',
            message: event.message || "Unknown error",
            data: {
                stack: event.error ? event.error.stack : null,
                url: window.location.href
            }
        })
    }).catch(() => {});
});

(function() {
    function escapeHtml(text) {
        if (!text) return '';
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    const input = document.getElementById('chat-input');
    const submitBtn = document.getElementById('chat-submit');
    const stopBtn = document.getElementById('chat-stop');
    const history = document.getElementById('chat-history');

    // base.html loads this file on every page for the global error logger above,
    // but the chat UI itself only exists on the chat view. A page with none of
    // it is the normal case (dashboard, portfolio, settings, ...) — bail quietly.
    // A page with *some* of it is a real breakage, so keep reporting that.
    if (!input && !submitBtn && !history) return;

    if (!input || !submitBtn || !history) {
        const errorMsg = "Chat JS Initialization Failed: Missing DOM Elements";
        console.error(errorMsg, { input: !!input, submitBtn: !!submitBtn, history: !!history });
        fetch('/api/logs/frontend', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                level: 'error',
                phase: 'UI',
                message: errorMsg,
                data: { 
                    input: !!input, 
                    submitBtn: !!submitBtn, 
                    history: !!history,
                    url: window.location.href
                }
            })
        }).catch(() => {});
        return;
    }

    let currentAbortController = null;
    let pendingAttachments = [];

    // --- Attachment Helper Functions ---
    function addAttachment(file) {
        if (pendingAttachments.some(att => att.name === file.name && att.size === file.size)) {
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            pendingAttachments.push({
                name: file.name,
                type: file.type || 'application/octet-stream',
                size: file.size,
                data: e.target.result
            });
            renderAttachmentPreviews();
        };
        reader.readAsDataURL(file);
    }

    function removeAttachment(index) {
        pendingAttachments.splice(index, 1);
        renderAttachmentPreviews();
    }

    function renderAttachmentPreviews() {
        const container = document.getElementById('attachment-preview-container');
        if (!container) return;

        container.innerHTML = '';

        if (pendingAttachments.length === 0) {
            container.classList.add('hidden');
            return;
        }

        container.classList.remove('hidden');

        pendingAttachments.forEach((att, idx) => {
            const chip = document.createElement('div');
            chip.className = 'flex items-center gap-2 px-3 py-1.5 bg-surface-container rounded-lg border border-outline-variant/30 text-xs text-on-surface';

            if (att.type.startsWith('image/')) {
                const thumb = document.createElement('img');
                thumb.src = att.data;
                thumb.className = 'w-6 h-6 object-cover rounded border border-outline-variant/30';
                chip.appendChild(thumb);
            } else {
                const icon = document.createElement('span');
                icon.className = 'material-symbols-outlined text-sm text-primary';
                icon.innerText = att.type === 'application/pdf' ? 'picture_as_pdf' : 'description';
                chip.appendChild(icon);
            }

            const nameText = document.createElement('span');
            nameText.innerText = att.name;
            nameText.className = 'max-w-[120px] truncate font-medium';
            chip.appendChild(nameText);

            const deleteBtn = document.createElement('button');
            deleteBtn.type = 'button';
            deleteBtn.className = 'text-on-surface-variant/60 hover:text-error transition-colors flex items-center justify-center ml-1';
            deleteBtn.innerHTML = '<span class="material-symbols-outlined text-xs">close</span>';
            deleteBtn.addEventListener('click', () => removeAttachment(idx));
            chip.appendChild(deleteBtn);

            container.appendChild(chip);
        });
    }

    // Attach button and File input listener
    const fileInput = document.getElementById('file-input');
    const attachBtn = document.getElementById('chat-attach');

    if (attachBtn && fileInput) {
        attachBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => {
            const files = e.target.files;
            if (files) {
                for (let i = 0; i < files.length; i++) {
                    addAttachment(files[i]);
                }
            }
            fileInput.value = '';
        });
    }

    // Clipboard paste listener
    if (input) {
        input.addEventListener('paste', (e) => {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let i = 0; i < items.length; i++) {
                if (items[i].type.indexOf('image') !== -1) {
                    const file = items[i].getAsFile();
                    if (file) {
                        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                        const pastedFile = new File([file], `pasted_image_${timestamp}.png`, { type: file.type });
                        addAttachment(pastedFile);
                    }
                }
            }
        });
    }

    // ── Deep / Ghost Toggle State ──
    let isDeepMode = false;
    let isGhostMode = false;
    const toggleDeep = document.getElementById('toggle-deep');
    const toggleGhost = document.getElementById('toggle-ghost');

    if (toggleDeep) {
        toggleDeep.addEventListener('click', () => {
            isDeepMode = !isDeepMode;
            if (isDeepMode) {
                toggleDeep.classList.remove('text-on-surface/30', 'border-outline-variant/20');
                toggleDeep.classList.add('text-primary', 'border-primary', 'bg-primary/10', 'shadow-[0_0_10px_rgba(78,222,163,0.3)]');
            } else {
                toggleDeep.classList.add('text-on-surface/30', 'border-outline-variant/20');
                toggleDeep.classList.remove('text-primary', 'border-primary', 'bg-primary/10', 'shadow-[0_0_10px_rgba(78,222,163,0.3)]');
            }
        });
    }

    if (toggleGhost) {
        toggleGhost.addEventListener('click', () => {
            isGhostMode = !isGhostMode;
            if (isGhostMode) {
                toggleGhost.classList.remove('text-on-surface/30', 'border-outline-variant/20');
                toggleGhost.classList.add('text-amber-400', 'border-amber-400', 'bg-amber-400/10', 'shadow-[0_0_10px_rgba(251,191,36,0.3)]');
            } else {
                toggleGhost.classList.add('text-on-surface/30', 'border-outline-variant/20');
                toggleGhost.classList.remove('text-amber-400', 'border-amber-400', 'bg-amber-400/10', 'shadow-[0_0_10px_rgba(251,191,36,0.3)]');
            }
        });
    }

    // ── Run Notices (console output from the last agent run) ──
    function updateRunNotices(notices) {
        const bar      = document.getElementById('run-notices-bar');
        const btn      = document.getElementById('run-notices-btn');
        const icon     = document.getElementById('run-notices-icon');
        const countEl  = document.getElementById('run-notices-count');
        const list     = document.getElementById('run-notices-list');
        if (!bar || !list) return;

        const hasWarn = notices.some(n => /[⚠❌🔴]/.test(n));

        list.innerHTML = notices.map(n => {
            const safe = n.replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const isWarn = /[⚠❌🔴]/.test(n);
            const cls = isWarn ? 'text-amber-400/90' : 'text-on-surface-variant/60';
            return `<div class="py-px ${cls}">${safe}</div>`;
        }).join('');

        icon.textContent = hasWarn ? '⚠' : '✓';
        countEl.textContent = notices.length;

        if (hasWarn) {
            btn.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 transition-colors';
        } else {
            btn.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-primary/10 border border-primary/20 text-primary/70 hover:bg-primary/20 transition-colors';
        }

        bar.classList.remove('hidden');
    }

    function clearRunNotices() {
        const bar = document.getElementById('run-notices-bar');
        const pop = document.getElementById('run-notices-popover');
        if (bar) bar.classList.add('hidden');
        if (pop) pop.classList.add('hidden');
    }

    // Notices pill toggle + dismiss
    const _noticesBtn     = document.getElementById('run-notices-btn');
    const _noticesPop     = document.getElementById('run-notices-popover');
    const _noticesDismiss = document.getElementById('run-notices-dismiss');

    if (_noticesBtn && _noticesPop) {
        _noticesBtn.addEventListener('click', () => _noticesPop.classList.toggle('hidden'));
    }
    if (_noticesDismiss) {
        _noticesDismiss.addEventListener('click', e => {
            e.stopPropagation();
            clearRunNotices();
        });
    }
    document.addEventListener('click', e => {
        if (_noticesPop && !_noticesPop.classList.contains('hidden')) {
            const bar = document.getElementById('run-notices-bar');
            if (bar && !bar.contains(e.target)) _noticesPop.classList.add('hidden');
        }
    });

    // ── Header Status (brief phase label) ──
    function updateHeaderStatus(fullStatus, visible = true) {
        const bar = document.getElementById('live-reasoning-bar');
        const text = document.getElementById('live-reasoning-text');
        if (!bar || !text) return;

        if (visible && fullStatus) {
            // Phase only — the header reflects WHAT the run is doing, not transient
            // failures. A recoverable mid-run hiccup (e.g. a retried planner cycle)
            // must NOT flash "ERROR" here, or the header churns ERROR→REASONING→
            // response. Terminal errors surface as the persistent in-chat error
            // card + the notices pill instead. ('plan' so "Planner …" maps here.)
            let brief = 'PROCESSING';
            const s = fullStatus.toLowerCase();
            if (s.includes('deep reasoning') || s.includes('plan'))       brief = 'REASONING';
            else if (s.includes('tool') || s.includes('executing'))       brief = 'TOOL EXEC';
            else if (s.includes('analyz') || s.includes('scanning'))      brief = 'ANALYZING';
            else if (s.includes('fetch') || s.includes('retriev'))        brief = 'FETCHING DATA';
            else if (s.includes('generat') || s.includes('compos'))       brief = 'COMPOSING';
            else if (s.includes('search'))                                brief = 'SEARCHING';

            text.innerText = brief;
            // Clear any leftover FAILED styling from a previous run (setHeaderError).
            text.style.color = '';
            const icon = bar.querySelector('.material-symbols-outlined');
            if (icon) { icon.style.color = ''; icon.classList.add('reasoning-pulse'); icon.textContent = 'psychology'; }
            bar.classList.remove('hidden');
            bar.classList.add('flex');
        } else {
            bar.classList.add('hidden');
            bar.classList.remove('flex');
        }
    }

    // ── Send / Stop Toggle ──
    // --- Live run pill ------------------------------------------------------
    // A fallback for the in-transcript trace panel, shown only when that panel
    // cannot do its job. The panel is pinned ABOVE the answer, so it scrolls off
    // as the response streams and leaves nothing on screen to say the run is
    // still going. But while it IS on screen it already carries the phase, the
    // timeline and its own spinner — a second box below would just repeat it.
    //
    // So the pill needs two conditions, not one: a run in flight (the stop
    // button's lifecycle) AND the panel out of view. What is left is only what
    // the panel cannot say from off screen: still running, and for how long —
    // the clock is self-driven, so the long silent stretches between status
    // events (a planner cycle can run 60s+ without emitting one) still visibly
    // tick instead of reading as a hang. The phase text moves to the tooltip:
    // available on hover, zero pixels when not.
    const runDock = document.getElementById('run-activity-dock');
    const runDockElapsed = document.getElementById('run-activity-elapsed');
    let runDockTimer = null;
    let runDockStarted = 0;
    let runInFlight = false;

    // Is this run's trace panel (the newest one) inside the transcript viewport?
    // Measured geometrically rather than with an IntersectionObserver: the panel
    // also leaves the viewport WITHOUT any scroll event, simply by being pushed
    // up as the answer streams in below it, and one rect comparison covers both
    // causes. It runs only while a run is in flight — on scroll and on the
    // clock's own tick — so it is a handful of reads per second, not a loop.
    function runPanelOnScreen() {
        const panels = history.querySelectorAll('.ai-status-area');
        const panel = panels[panels.length - 1];
        if (!panel) return false;   // nothing to defer to → the pill is the only signal left
        const p = panel.getBoundingClientRect();
        const h = history.getBoundingClientRect();
        // Enough of it on screen to actually read the live summary line, not a
        // one-pixel sliver at the edge — a sliver is not a progress indicator,
        // and treating it as one would suppress the pill exactly when the panel
        // is halfway out the top.
        return Math.min(p.bottom, h.bottom) - Math.max(p.top, h.top) >= 24;
    }

    function syncRunDock() {
        if (!runDock) return;
        const show = runInFlight && !runPanelOnScreen();
        runDock.classList.toggle('hidden', !show);
        runDock.classList.toggle('flex', show);
    }

    // title, not textContent: the phase is worth keeping, but not worth a line
    // of chrome next to a panel that already prints it.
    function setRunDockStatus(text) {
        if (runDock && text) runDock.title = text;
    }

    // Scrolling is the main way the panel leaves the viewport. Throttled on the
    // clock rather than rAF: rAF is tied to the rendering step and does not run
    // in a backgrounded or non-painting tab, which would strand the pill in
    // whatever state it was last left in. The 1s tick is the trailing
    // correction, so a scroll that ends inside the throttle window still
    // settles.
    let runDockScrollTs = 0;
    history.addEventListener('scroll', () => {
        if (!runInFlight || Date.now() - runDockScrollTs < 100) return;
        runDockScrollTs = Date.now();
        syncRunDock();
    }, { passive: true });

    function showRunDock() {
        if (!runDock) return;
        runInFlight = true;
        runDockStarted = Date.now();
        setRunDockStatus('Initializing Intelligence...');
        if (runDockElapsed) runDockElapsed.textContent = '0s';
        syncRunDock();
        clearInterval(runDockTimer);
        // The clock tick doubles as the visibility poll, which is what catches
        // the panel drifting off screen under a streaming answer.
        runDockTimer = setInterval(() => {
            if (runDockElapsed) {
                runDockElapsed.textContent = `${Math.round((Date.now() - runDockStarted) / 1000)}s`;
            }
            syncRunDock();
        }, 1000);
    }

    function hideRunDock() {
        clearInterval(runDockTimer);
        runDockTimer = null;
        runInFlight = false;
        syncRunDock();
    }

    // Click → back to the live trace. The pill is deliberately contentless; the
    // panel it points at is the full timeline, and this is how you get there
    // without hunting for it in the scrollback.
    if (runDock) {
        runDock.addEventListener('click', () => {
            const panels = history.querySelectorAll('.ai-status-area');
            const panel = panels[panels.length - 1];
            if (!panel) return;
            panel.open = true;
            panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });
    }

    // The dock shares the stop button's lifecycle: both mean "a run is in
    // flight", and every start/abort/error/completion path already routes
    // through this pair.
    function showStopButton() {
        if (stopBtn) stopBtn.style.display = 'flex';
        if (submitBtn) submitBtn.style.display = 'none';
        showRunDock();
    }

    function hideStopButton() {
        if (stopBtn) stopBtn.style.display = 'none';
        if (submitBtn) submitBtn.style.display = 'flex';
        hideRunDock();
    }

    // Wire up stop button
    if (stopBtn) {
        stopBtn.addEventListener('click', () => {
            if (currentAbortController) {
                currentAbortController.abort();
                updateHeaderStatus(null, false);
                hideStopButton();
            }
        });
    }

    async function sendMessage() {
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }

        clearRunNotices();

        // Check if there's a full prompt stored (from quick action buttons)
        const fullPrompt = input.getAttribute('data-full-prompt');
        const text = fullPrompt || input.value.trim();
        const displayText = input.value.trim(); // What user sees
        
        if (!text && pendingAttachments.length === 0) return;

        const scrollContainer = history;

        // Persistent, plain-English error surface so the user never needs the
        // console/logs. The transient status line is removed at run end; this
        // card stays in the transcript. One card per run (fresh closure each send).
        let errorCardShown = false;
        function briefError(err) {
            const s = ((err && (err.message || err)) || '').toString().toLowerCase();
            if (s.includes('deploymentnotfound') || s.includes('does not exist'))
                return 'AI model deployment not found — check the model settings.';
            if (s.includes('429') || s.includes('rate limit') || s.includes('throttl') || s.includes('quota'))
                return 'The AI service is rate-limited right now. Wait a moment and retry.';
            if (s.includes('401') || s.includes('unauthorized') || s.includes('api key'))
                return 'Authentication failed — check your API key in Settings.';
            if (s.includes('timeout') || s.includes('timed out'))
                return 'The request timed out. Please try again.';
            if (s.includes('failed to fetch') || s.includes('networkerror') || s.includes('connection') ||
                s.includes(' 500') || s.includes(' 502') || s.includes(' 503'))
                return 'Could not reach the server. Check it is running and retry.';
            return 'An unexpected error occurred. Please retry.';
        }
        function appendErrorCard(brief) {
            if (errorCardShown) return;
            errorCardShown = true;
            const safe = String(brief == null ? '' : brief).replace(/</g, '&lt;').replace(/>/g, '&gt;');
            const card = document.createElement('div');
            card.className = 'flex gap-6 py-2';
            card.innerHTML = `
                <div class="w-10 h-10 flex-shrink-0"></div>
                <div class="flex-1 flex items-start gap-2 rounded-xl border border-error/30 bg-error/10 px-3 py-2">
                    <span class="material-symbols-outlined text-error text-base">error</span>
                    <div>
                        <p class="text-[11px] font-bold uppercase tracking-widest text-error">Something went wrong</p>
                        <p class="text-[12px] text-error/80 font-body mt-0.5">${safe}</p>
                    </div>
                </div>
            `;
            history.appendChild(card);
            scrollContainer.scrollTop = scrollContainer.scrollHeight;
        }

        // PERMANENT terminal header states. Unlike the live phase brief (hidden
        // when a run ends), these stay put until the next run resets them, so the
        // user knows the result is incomplete/degraded without watching logs.
        //   runFailed   → FAILED (red):    the run did not recover (no usable answer)
        //   runDegraded → DEGRADED (amber): the run completed but the backend
        //                 flagged a deliberate degradation via send_status(...,
        //                 degraded=True) → payload.degraded (e.g. a planner cycle
        //                 was rate-limited), so the answer may be missing data.
        //                 Driven only by that explicit flag — NOT by emoji in the
        //                 status text or by benign notices. FAILED beats DEGRADED.
        let runFailed = false;
        let runDegraded = false;
        function _setHeaderTerminal(label, color, iconName) {
            const bar = document.getElementById('live-reasoning-bar');
            const text = document.getElementById('live-reasoning-text');
            if (!bar || !text) return;
            const icon = bar.querySelector('.material-symbols-outlined');
            text.innerText = label;
            text.style.color = color;
            if (icon) { icon.style.color = color; icon.classList.remove('reasoning-pulse'); icon.textContent = iconName; }
            bar.classList.remove('hidden');
            bar.classList.add('flex');
        }
        function setHeaderError()   { _setHeaderTerminal('FAILED', '#f87171', 'error'); }
        function setHeaderDegraded() { _setHeaderTerminal('DEGRADED', '#fbbf24', 'warning'); }
        // A clean run used to end by simply HIDING the pill. An absence is not a
        // signal: a finished run and a run whose header quietly stopped updating
        // looked identical. Say COMPLETE explicitly, then clear it after a few
        // seconds so a stale badge doesn't carry into the next question.
        function setHeaderComplete() {
            _setHeaderTerminal('COMPLETE', '#4ade80', 'check_circle');
            setTimeout(() => {
                const t = document.getElementById('live-reasoning-text');
                // Only clear if no newer run has claimed the header since.
                if (t && t.innerText === 'COMPLETE') updateHeaderStatus(null, false);
            }, 4000);
        }

        input.value = '';
        input.removeAttribute('data-full-prompt'); // Clear the stored prompt

        const attachmentsToSend = [...pendingAttachments];
        pendingAttachments = [];
        renderAttachmentPreviews();

        let attachmentHtml = '';
        if (attachmentsToSend.length > 0) {
            attachmentHtml = '<div class="flex flex-wrap gap-2 mt-3 pt-3 border-t border-outline-variant/10">';
            attachmentsToSend.forEach(att => {
                const escapedName = escapeHtml(att.name);
                if (att.type.startsWith('image/')) {
                    attachmentHtml += `
                        <div class="relative group rounded-lg overflow-hidden border border-outline-variant/30 max-w-[200px] bg-surface-container-low">
                            <img src="${att.data}" class="max-h-32 object-contain" />
                            <div class="absolute bottom-0 left-0 w-full px-2 py-1 bg-black/60 text-[9px] text-on-surface truncate">${escapedName}</div>
                        </div>
                    `;
                } else {
                    attachmentHtml += `
                        <div class="flex items-center gap-2 px-3 py-1.5 bg-surface-container rounded-lg border border-outline-variant/20 text-xs">
                            <span class="material-symbols-outlined text-sm text-primary">${att.type === 'application/pdf' ? 'picture_as_pdf' : 'description'}</span>
                            <span class="max-w-[120px] truncate font-medium text-on-surface-variant">${escapedName}</span>
                        </div>
                    `;
                }
            });
            attachmentHtml += '</div>';
        }

        const escapedDisplayText = escapeHtml(displayText);

        // 1. Append User Message
        const userDiv = document.createElement('div');
        userDiv.className = 'flex gap-6 flex-row-reverse';
        userDiv.innerHTML = `
            <div class="w-10 h-10 rounded-xl bg-surface-variant border border-outline-variant/30 flex-shrink-0 flex items-center justify-center">
                <span class="material-symbols-outlined text-on-surface-variant text-xl">person</span>
            </div>
            <div class="flex-1 flex flex-col items-end">
                <div class="flex items-center gap-3 mb-2">
                    <span class="text-[10px] text-on-surface-variant/40">JUST NOW</span>
                    <span class="font-headline font-extrabold text-xs text-on-surface uppercase tracking-widest">Architect</span>
                </div>
                <div class="bg-surface-container-high border border-outline-variant/20 px-6 py-4 rounded-2xl rounded-tr-none max-w-xl shadow-xl">
                    <p class="text-on-surface text-[15px] leading-relaxed font-medium whitespace-pre-wrap break-words">${escapedDisplayText}</p>
                    ${attachmentHtml}
                </div>
            </div>
        `;
        history.appendChild(userDiv);
        scrollContainer.scrollTop = scrollContainer.scrollHeight;

        // 2. Create AI response container immediately
        const aiDivOuter = document.createElement('div');
        aiDivOuter.className = 'flex gap-6';
        aiDivOuter.innerHTML = `
            <div class="w-10 h-10 rounded-xl bg-primary-container border border-primary/20 flex-shrink-0 flex items-center justify-center shadow-lg shadow-primary/5">
                <span class="material-symbols-outlined text-primary text-xl">account_tree</span>
            </div>
        `;

        const contentCont = document.createElement('div');
        contentCont.className = 'flex-1 space-y-4';
        contentCont.innerHTML = `
            <div class="flex items-center gap-3">
                <span class="font-headline font-extrabold text-xs text-primary uppercase tracking-widest">CairnIQ</span>
                <span class="text-[10px] text-on-surface-variant/40">JUST NOW</span>
            </div>
        `;

        const markdownCont = document.createElement('div');
        markdownCont.className = 'text-on-surface-variant text-base leading-relaxed markdown-content';

        // ONE live surface for the whole run: the phase line is the header, the
        // model's reasoning is the body. These used to be two elements — a
        // spinner reading "Initializing Intelligence..." and a separate trace
        // panel — which split them exactly wrong: the empty one occupied the
        // long pre-answer gap (median ~33s) while the one with real content
        // only appeared once reasoning arrived, often half a minute in.
        //
        // Present from the moment the run starts, so the box fills in rather
        // than replacing a spinner.
        const statusArea = document.createElement('details');
        statusArea.className = 'ai-status-area agent-thinking-log rounded-xl border border-outline-variant/20 bg-surface-container-low/50 px-3 py-2';
        statusArea.open = true;
        statusArea.innerHTML = `
            <summary class="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant/50 cursor-pointer select-none list-none flex items-center gap-2">
                <div class="status-spinner w-3 h-3 rounded-full border-2 border-primary border-t-transparent animate-spin shrink-0"></div>
                <span class="status-text text-primary normal-case tracking-normal font-body italic font-normal">Initializing Intelligence...</span>
                <span class="trace-size font-normal normal-case tracking-normal opacity-70 ml-auto shrink-0"></span>
            </summary>
            <div class="trace-content mt-2 text-[11px] leading-relaxed text-on-surface-variant/60 font-body break-words h-72 min-h-[6rem] overflow-y-auto resize-y"></div>
        `;
        const traceEl = statusArea.querySelector('.trace-content');
        traceEl.textContent = 'Waiting for the model to start reasoning…';

        // Declared out here, not inside the stream try-block: finishRunPanel is
        // defined at this scope and needs to know what arrived.
        let thinkingText = "";
        let lastActivity = null;

        // What did the BROWSER actually receive? The server logs what it queued
        // and what it yielded, but until now nothing recorded what arrived at
        // the other end, so "the panel is empty" could mean a dead channel or a
        // silent provider and there was no way to tell them apart from either
        // side alone. Reported once per run; failures here never affect the run.
        const wireStats = { started: Date.now(), firstStatusMs: null, firstThinkingMs: null,
                            firstTextMs: null, status: 0, heartbeat: 0, thinking: 0, text: 0, other: 0 };
        function noteWire(kind) {
            const t = Date.now() - wireStats.started;
            if (kind === 'status' && wireStats.firstStatusMs === null) wireStats.firstStatusMs = t;
            if (kind === 'thinking' && wireStats.firstThinkingMs === null) wireStats.firstThinkingMs = t;
            if (kind === 'text' && wireStats.firstTextMs === null) wireStats.firstTextMs = t;
            wireStats[kind] = (wireStats[kind] || 0) + 1;
        }
        function reportWire(outcome) {
            try {
                fetch('/api/logs/frontend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        level: 'info', phase: 'Stream',
                        message: 'Client stream summary',
                        data: { ...wireStats, outcome, durationMs: Date.now() - wireStats.started,
                                steps: traceEl.querySelectorAll('.trace-step').length,
                                reasoningChars: thinkingText.length }
                    })
                }).catch(() => {});
            } catch (e) { /* telemetry must never break a run */ }
        }

        // The body is a chronological ACTIVITY LOG, not just a reasoning dump.
        // Reasoning can be minutes away (or absent entirely), but status events
        // land from the first few milliseconds — routing, tool starts, tool
        // finishes. Showing only reasoning left a big box reading "waiting" while
        // a perfectly good progress feed was being thrown away into a one-line
        // header. Both go here, interleaved in the order they happened.
        function clearPlaceholder() {
            if (traceEl.dataset.filled !== '1') {
                traceEl.textContent = '';
                traceEl.dataset.filled = '1';
            }
        }

        function appendActivity(text) {
            // Consecutive duplicates are common (a phase re-announced by two
            // nodes) and add nothing to a timeline.
            if (!text || text === lastActivity) return;
            lastActivity = text;
            clearPlaceholder();
            const row = document.createElement('div');
            row.className = 'trace-step py-px text-on-surface-variant/80';
            row.textContent = text;           // textContent: status text is data
            traceEl.appendChild(row);
            traceEl.scrollTop = traceEl.scrollHeight;
        }

        function appendReasoning(text) {
            if (!text) return;
            clearPlaceholder();
            // Keep appending into the current reasoning block so a paragraph
            // reads as prose, but start a new one after an intervening status
            // event so the interleaving stays chronological.
            let block = traceEl.lastElementChild;
            if (!block || !block.classList.contains('trace-reasoning')) {
                block = document.createElement('div');
                block.className = 'trace-reasoning whitespace-pre-wrap my-1 pl-2 border-l-2 border-primary/30 opacity-90';
                traceEl.appendChild(block);
            }
            block.textContent += text;        // textContent: raw model output
            traceEl.scrollTop = traceEl.scrollHeight;
        }

        // Set the moment the first answer token lands. That token is the honest
        // boundary between "still working" and "answering" — before it, the panel
        // is live; after it, it is a record. The spinner used to ignore this
        // boundary and keep turning until the stream closed, so the whole answer
        // streamed in under a header still claiming to be thinking.
        let answerStarted = false;

        // Swap the indeterminate spinner for a settled state icon.
        function _settleSpinner(iconName, colorClass) {
            const spinner = statusArea.querySelector('.status-spinner');
            if (!spinner) return;              // already settled
            const icon = document.createElement('span');
            icon.className = `material-symbols-outlined text-sm shrink-0 leading-none ${colorClass}`;
            icon.textContent = iconName;
            spinner.replaceWith(icon);
        }

        // Stamp the summary line so a settled panel reads as a closed record
        // ("4,312 chars · 41s") rather than a live one that stopped moving.
        function _stampPanelSize() {
            const sizeEl = statusArea.querySelector('.trace-size');
            if (!sizeEl) return;
            const secs = Math.round((Date.now() - wireStats.started) / 1000);
            const chars = thinkingText ? `${thinkingText.length.toLocaleString()} chars · ` : '';
            sizeEl.textContent = `${chars}${secs}s`;
        }

        // First answer token → the thinking phase is over. Settle the panel and
        // collapse it: the trace has had the top of the view for the whole
        // pre-answer gap and must now yield it to the answer. It is collapsed,
        // not removed — the summary line names what it holds and one click
        // reopens it. (Collapsing was previously avoided because the collapsed
        // strip was unlabelled; it now carries label, size and duration.)
        function markReasoningDone() {
            if (answerStarted || !statusArea.parentNode) return;
            answerStarted = true;
            _settleSpinner('check_circle', 'text-primary');
            const label = statusArea.querySelector('.status-text');
            if (label) {
                label.textContent = thinkingText ? 'Reasoning trace' : 'Run activity';
                label.classList.remove('italic');
            }
            _stampPanelSize();
            statusArea.open = false;
            updateHeaderStatus('composing response');
            setRunDockStatus('Composing response...');
        }

        // End-of-run teardown for the live panel. Settles the spinner, freezes the
        // header, and — when the run returned no reasoning at all — says so
        // explicitly. A silently blank body would otherwise look like the panel
        // had failed, when the honest answer is that the provider returned
        // nothing (e.g. reasoning effort off, or a path that never streams).
        function finishRunPanel() {
            if (!statusArea.parentNode) return;
            // No-op when markReasoningDone already settled it mid-stream.
            _settleSpinner(runFailed ? 'error' : 'check_circle',
                           runFailed ? 'text-error' : 'text-primary');
            const label = statusArea.querySelector('.status-text');
            if (label) {
                // Name it for what it actually holds: a run that logged steps but
                // got no reasoning back still has a useful timeline in there.
                label.textContent = thinkingText ? 'Reasoning trace' : 'Run activity';
                label.classList.remove('italic');
            }
            _stampPanelSize();
            if (traceEl.dataset.filled !== '1') {
                traceEl.textContent = 'No activity or reasoning was reported for this run.';
                statusArea.open = false;
            }
        }

        // Reasoning above the answer: it happens first, and during the pre-answer
        // gap there is no answer yet, so the live box sits right where the user
        // is already looking.
        contentCont.appendChild(statusArea);
        contentCont.appendChild(markdownCont);
        aiDivOuter.appendChild(contentCont);
        history.appendChild(aiDivOuter);
        scrollContainer.scrollTop = scrollContainer.scrollHeight;

        // 3. Initialize
        updateHeaderStatus('Initializing Intelligence...', true);
        const controller = new AbortController();
        currentAbortController = controller;
        showStopButton();

        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: text,
                    deep: isDeepMode,
                    ghost: isGhostMode,
                    thread_id: window._activeThreadId || null,
                    attachments: attachmentsToSend,
                    // This client understands {delta} frames; ask for them.
                    stream_deltas: true
                }),
                signal: controller.signal
            });

            if (!res.ok) {
                const errorData = await res.text();
                throw new Error(`Server returned ${res.status}: ${errorData}`);
            }

            const threadIdFromHeader = res.headers.get('X-Thread-ID');
            if (threadIdFromHeader) {
                window._activeThreadId = threadIdFromHeader;
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder("utf-8");
            let buffer = "";

            let pendingText = "";
            let renderPending = false;
            // The answer assembled client-side from {delta} frames. The server
            // used to re-send the entire answer on every token, which is O(n^2)
            // on the wire (~50MB for a 20,000-char answer); we now accumulate
            // here and the server sends only what is new.
            let answerText = "";

            function queueRender(text) {
                pendingText = text;
                if (renderPending) return;

                renderPending = true;
                requestAnimationFrame(() => {
                    // Keep statusArea in place — it sits ABOVE the answer and is
                    // settled by markReasoningDone() on the first token, then
                    // finalized at stream end (success/abort/error paths).

                    // Check scroll position BEFORE updating DOM to prevent height-change bugs
                    const isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 50;

                    if (typeof marked !== 'undefined') {
                        let parsedHtml = marked.parse(pendingText);
                        markdownCont.innerHTML = parsedHtml.replace(/<p>\s*<\/p>/g, '');
                    } else {
                        markdownCont.innerText = pendingText;
                    }

                    if (isAtBottom) {
                        scrollContainer.scrollTop = scrollContainer.scrollHeight;
                    }
                    renderPending = false;
                });
            }

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.trim() === '') continue;
                    try {
                        const payload = JSON.parse(line);

                        // Capture thread_id to persist session continuity
                        if (payload.thread_id) {
                            window._activeThreadId = payload.thread_id;
                        }

                        // Status updates → in-chat detail + header brief. These are
                        // a live, transient feed (a "❌ cycle failed" can be followed
                        // by recovery), so keep the in-progress spinner + "..." here;
                        // terminal errors are shown by the persistent error card.
                        if (payload.status) {
                            const localStatus = statusArea.querySelector('.status-text');
                            // A heartbeat re-sends the CURRENT phase with an elapsed
                            // counter because the agent emits no status during a long
                            // LLM call. Render the counter instead of the trailing
                            // "..." so a frozen-looking line still visibly ticks.
                            // Only while the panel is still live. Once the answer
                            // is streaming the summary line is a settled label
                            // ("Reasoning trace · 41s"); a late status must not
                            // overwrite it back into an in-progress "…" phase.
                            if (localStatus && !answerStarted) {
                                localStatus.innerText = payload.heartbeat
                                    ? `${payload.status} · ${payload.elapsed}s`
                                    : payload.status + '...';
                            }
                            // A degrading event mid-run (e.g. a rate-limited planner
                            // cycle) is flagged EXPLICITLY by the backend via
                            // send_status(..., degraded=True) → payload.degraded. We
                            // only SHOW it at run-end (as DEGRADED) and only if the run
                            // wasn't FAILED, so flagging here is correct even when the
                            // run later recovers. Driven off the explicit flag rather
                            // than regex-matching ⚠️/❌ glyphs in the text, so a benign
                            // status that uses a glyph decoratively (e.g. "🚩 N tickers
                            // flagged") no longer false-positives the badge.
                            if (payload.degraded === true) runDegraded = true;
                            noteWire(payload.heartbeat ? 'heartbeat' : 'status');
                            // A heartbeat re-sends whatever phase the backend is
                            // still nominally in, so once text is flowing it would
                            // flip the header COMPOSING → REASONING and back. Real
                            // phase changes still update it.
                            if (!(answerStarted && payload.heartbeat)) updateHeaderStatus(payload.status);
                            // The pill's tooltip takes heartbeats too: it costs no
                            // space, and a hover mid-run should name the phase the
                            // agent is actually in. Bare status, no elapsed — the
                            // pill runs its own clock.
                            if (!answerStarted) setRunDockStatus(payload.status);
                            // Real phase changes become timeline entries in the
                            // panel body. Heartbeats are excluded on purpose:
                            // they re-send the CURRENT phase every 10s, so
                            // logging them would bury the timeline in repeats.
                            if (!payload.heartbeat) appendActivity(payload.status);
                        }

                        // Console notices from the run → informational header pill ONLY.
                        // These are NOT used to flag DEGRADED. ctx.notices collects any
                        // ⚠️/Failed/Error line, including benign events that don't make
                        // the answer incomplete — an FMP→web-search fallback, a BM25/FAISS
                        // init warning, or a streaming call that recovered via invoke.
                        // Driving the badge off these falsely degraded clean runs. DEGRADED
                        // is now driven solely by the explicit payload.degraded flag
                        // (above) — a genuine planner failure sets it via send_status.
                        if (payload.notices && payload.notices.length > 0) {
                            updateRunNotices(payload.notices);
                        }

                        // Terminal, unrecovered failure signalled by the backend:
                        // mark the run failed, show the persistent card. The header
                        // is set to its permanent FAILED state once the stream ends.
                        if (payload.fatal_error) {
                            runFailed = true;
                            appendErrorCard(briefError(payload.fatal_error));
                        }

                        // Live reasoning — fills the body of the run panel.
                        // textContent (not innerHTML) because this is raw model
                        // output. Auto-scrolls so the newest reasoning is in view.
                        if (payload.thinking) {
                            noteWire('thinking');
                            thinkingText += payload.thinking;
                            appendReasoning(payload.thinking);
                            // Size hint on the header, so a collapsed panel still
                            // reads as "there is something in here".
                            const sizeEl = statusArea.querySelector('.trace-size');
                            if (sizeEl) sizeEl.textContent = `${thinkingText.length.toLocaleString()} chars`;
                        }

                        // AI answer text. Two shapes:
                        //   {delta}  incremental — append (the normal case)
                        //   {text}   authoritative full answer — replace
                        // The server sends `text` on the first frame, whenever a
                        // sanitizer retroactively rewrites what it already sent
                        // (a <thinking> or watch block completing), and once more
                        // at stream end to reconcile. Replace on `text` rather
                        // than append, or a resync would duplicate the answer.
                        if (payload.delta) {
                            noteWire('text');
                            markReasoningDone();
                            answerText += payload.delta;
                            queueRender(answerText);
                        }
                        if (payload.text) {
                            noteWire('text');
                            // The answer has begun — retire the live panel before
                            // rendering, so the box never streams under a spinner.
                            markReasoningDone();
                            answerText = payload.text;
                            queueRender(answerText);
                        }
                    } catch (e) { }
                }
            }

            // Finalize - ensure the last accumulated render is complete
            if (pendingText) {
                const isAtBottom = scrollContainer.scrollHeight - scrollContainer.scrollTop - scrollContainer.clientHeight < 50;
                if (typeof marked !== 'undefined') {
                    let parsedHtml = marked.parse(pendingText);
                    markdownCont.innerHTML = parsedHtml.replace(/<p>\s*<\/p>/g, '');
                } else {
                    markdownCont.innerText = pendingText;
                }
                if (isAtBottom) {
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                }
            }

            // The panel is retired, not removed: it carries the run's reasoning,
            // which is worth keeping in the transcript. Left EXPANDED on purpose
            // — auto-collapsing hid the reasoning behind a thin unlabelled strip
            // the moment it finished, which defeats the point of showing it.
            finishRunPanel();
            reportWire('completed');
            if (currentAbortController === controller) {
                // Leave a PERMANENT FAILED header if the run ended unrecovered;
                // otherwise hide the live brief as normal.
                if (runFailed) setHeaderError();
                else if (runDegraded) setHeaderDegraded();
                else setHeaderComplete();
                hideStopButton();
                currentAbortController = null;
            }

            // Refresh the sidebar and session cost after successful message streaming completion
            if (typeof window.loadSavedChats === 'function') {
                window.loadSavedChats();
            }
            if (typeof window.loadSessionCost === 'function') {
                window.loadSessionCost();
            }

            // The turn is complete and the backend has captured it into the
            // feedback store. Announce it so the rating control in the footer
            // can arm itself for THIS turn (roadmap 1.5). An event, not a direct
            // call: this file loads on every page, and only the chat page has
            // that control.
            document.dispatchEvent(new CustomEvent('cairniq:turn-complete', {
                detail: { threadId: window._activeThreadId || null }
            }));
        } catch (e) {
            // Same retire-don't-remove treatment on the failure/abort path: a run
            // that died partway is exactly when its reasoning is worth reading.
            finishRunPanel();
            reportWire(e && e.name === 'AbortError' ? 'aborted' : 'error');
            const isSelfAbort = currentAbortController === controller;
            if (isSelfAbort) {
                hideStopButton();
                currentAbortController = null;
            }

            if (e.name === 'AbortError') {
                // User cancelled — not a failure: hide the live brief.
                if (isSelfAbort) {
                    updateHeaderStatus(null, false);
                    const cancelDiv = document.createElement('div');
                    cancelDiv.className = 'flex gap-6 py-2';
                    cancelDiv.innerHTML = `
                        <div class="w-10 h-10 flex-shrink-0"></div>
                        <div class="flex-1 flex items-center gap-2 text-error/60">
                            <span class="material-symbols-outlined text-sm">cancel</span>
                            <p class="text-[11px] font-bold uppercase tracking-widest">Analysis Decommissioned</p>
                        </div>
                    `;
                    history.appendChild(cancelDiv);
                    scrollContainer.scrollTop = scrollContainer.scrollHeight;
                }
            } else {
                // Hard/transport failure (server down, network, 5xx): persistent
                // card + a permanent FAILED header so the user knows it didn't complete.
                console.error(e);
                appendErrorCard(briefError(e));
                if (isSelfAbort) setHeaderError();
            }
        }
    }

    // Input listeners
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    submitBtn.addEventListener('click', sendMessage);

    // Expose globally for quick-action buttons in index.html
    window.sendMessage = sendMessage;
    window.resetChatComposer = function() {
        pendingAttachments = [];
        input.value = '';
        input.removeAttribute('data-full-prompt');
        renderAttachmentPreviews();
    };
})();
