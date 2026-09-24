import { emptyProgress, filterQuestions, parseProgress, storageKey, summarize, type QuestionIndex, type Progress, type Rating } from '../utils/interview-state';

export function setupInterviewBanks() {
  document.querySelectorAll<HTMLElement>('[data-interview-bank]').forEach((bank) => {
    if (bank.dataset.ready) return;
    const one = <T extends HTMLElement = HTMLElement>(selector: string) => bank.querySelector<T>(selector)!;
    const cards = Array.from(bank.querySelectorAll<HTMLElement>('[data-question]'));
    if (!cards.length) return;
    bank.dataset.ready = 'true';
    const abort = new AbortController();
    const listen = (element: EventTarget, type: string, callback: EventListener) => element.addEventListener(type, callback, { signal: abort.signal });
    const index: QuestionIndex[] = cards.map((card) => ({ id: card.dataset.question!, topic: card.dataset.topic!, text: card.querySelector('.interview-prompt')?.textContent ?? '' }));
    const byId = new Map(cards.map((card) => [card.dataset.question!, card]));
    const validIds = new Set(byId.keys());
    const topic = one<HTMLSelectElement>('[data-topic-filter]');
    const search = one<HTMLInputElement>('[data-search]');
    let progress: Progress;
    let stored = true;
    try { progress = parseProgress(localStorage.getItem(storageKey), validIds); } catch { progress = emptyProgress(); stored = false; }
    let mode: 'read' | 'practice' = 'read';
    let current = progress.lastQuestion ?? index[0].id;
    let revealed = false;
    let finished = false;
    let expanded = false;
    let reviewQueue: string[] | null = null;
    let flipping = false;
    let transitionGeneration = 0;
    let animation: Animation | undefined;
    const announce = (message: string) => { one('[data-announcement]').textContent = message; };
    const save = () => {
      try { localStorage.setItem(storageKey, JSON.stringify(progress)); } catch { stored = false; }
      one('[data-storage-note]').textContent = stored ? 'Progress is saved in this browser.' : 'Browser storage is unavailable. Progress lasts for this visit.';
    };
    const matches = () => filterQuestions(index, topic.value, search.value);
    const queue = () => matches().map(({ id }) => id).filter((id) => reviewQueue === null || reviewQueue.includes(id));
    const closeDetails = () => byId.get(current)?.querySelectorAll<HTMLDetailsElement>('details').forEach((detail) => { detail.open = false; });
    const cancelFlip = () => { transitionGeneration++; animation?.cancel(); animation = undefined; flipping = false; one<HTMLButtonElement>('[data-reveal]').disabled = false; };
    const syncExpandLabel = () => {
      const visible = cards.filter((card) => !card.hidden);
      expanded = visible.length > 0 && visible.every((card) => card.querySelector<HTMLDetailsElement>('.interview-detail')!.open);
      one('[data-expand]').textContent = expanded ? 'Collapse all answers' : 'Expand all answers';
    };

    function update(focus = false) {
      const matched = matches();
      const ids = queue();
      if (!ids.includes(current)) { current = ids[0] ?? ''; revealed = false; closeDetails(); }
      bank.dataset.mode = mode;
      bank.dataset.revealed = String(revealed);
      const active = mode === 'practice' && ids.length > 0 && !finished;
      const matchSet = new Set(matched.map(({ id }) => id));
      cards.forEach((card) => {
        const id = card.dataset.question!;
        card.hidden = mode === 'read' ? !matchSet.has(id) : !active || id !== current;
        const hideAnswer = mode === 'practice' && !revealed;
        card.querySelectorAll<HTMLElement>('.interview-brief, .interview-detail, .interview-answer-kind').forEach((element) => { element.hidden = hideAnswer; });
      });
      bank.querySelectorAll<HTMLElement>('[data-topic-section]').forEach((section) => {
        section.hidden = !Array.from(section.querySelectorAll<HTMLElement>('[data-question]')).some((card) => !card.hidden);
      });
      bank.querySelectorAll<HTMLButtonElement>('[data-mode-button]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.modeButton === mode)));
      one('[data-empty]').hidden = matched.length > 0;
      one('[data-result-count]').textContent = `${matched.length} ${matched.length === 1 ? 'question' : 'questions'}`;
      one('[data-expand]').hidden = mode !== 'read';
      if (mode === 'read') syncExpandLabel();
      one('[data-practice-header]').hidden = !active;
      one('[data-practice-actions]').hidden = !active;
      one('[data-finish]').hidden = mode !== 'practice' || !finished;
      one('[data-storage]').hidden = mode !== 'practice';
      const count = summarize(ids, progress.ratings);
      one('[data-progress]').textContent = `${count.reviewed} reviewed · ${count.gotIt} Got it · ${count.again} Again`;
      one('[data-position]').textContent = `${ids.indexOf(current) + 1} / ${ids.length}`;
      one<HTMLButtonElement>('[data-previous]').disabled = ids.indexOf(current) <= 0;
      one('[data-next]').textContent = ids.indexOf(current) === ids.length - 1 ? 'Finish' : 'Next';
      one('[data-reveal]').hidden = false;
      one('[data-reveal]').textContent = revealed ? 'Answer shown' : 'Show answer';
      one('[data-reveal]').setAttribute('aria-disabled', String(revealed));
      one('[data-ratings]').hidden = !revealed;
      bank.querySelectorAll<HTMLButtonElement>('[data-rate]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.rate === progress.ratings[current])));
      one('[data-summary]').textContent = `${count.reviewed} reviewed: ${count.gotIt} Got it, ${count.again} Again. ${count.unreviewed} unreviewed.`;
      one<HTMLButtonElement>('[data-review-again]').disabled = count.again === 0;
      if (focus && active) byId.get(current)?.querySelector<HTMLElement>('.interview-prompt')?.focus({ preventScroll: true });
      if (active) { progress.lastQuestion = current; save(); }
    }

    const resetCard = () => { cancelFlip(); revealed = false; finished = false; closeDetails(); };
    bank.querySelectorAll<HTMLButtonElement>('[data-mode-button]').forEach((button) => listen(button, 'click', () => {
      resetCard(); mode = button.dataset.modeButton as typeof mode;
      update();
    }));
    const changeFilter = () => { resetCard(); reviewQueue = null; update(); };
    listen(topic, 'change', changeFilter); listen(search, 'input', changeFilter);
    bank.querySelectorAll('[data-clear]').forEach((button) => listen(button, 'click', () => { topic.value = ''; search.value = ''; changeFilter(); }));
    listen(one('[data-expand]'), 'click', () => {
      syncExpandLabel();
      expanded = !expanded;
      cards.filter((card) => !card.hidden).forEach((card) => { card.querySelector<HTMLDetailsElement>('.interview-detail')!.open = expanded; });
      update();
    });
    let disclosureFrame = 0;
    bank.addEventListener('toggle', (event) => {
      if (mode !== 'read' || !(event.target as Element).matches('.interview-detail')) return;
      cancelAnimationFrame(disclosureFrame);
      disclosureFrame = requestAnimationFrame(syncExpandLabel);
    }, { capture: true, signal: abort.signal });
    listen(one('[data-reveal]'), 'click', async () => {
      if (flipping || revealed || !current) return;
      flipping = true;
      const generation = ++transitionGeneration;
      const button = one<HTMLButtonElement>('[data-reveal]'); button.setAttribute('aria-disabled', 'true');
      const surface = byId.get(current)!.querySelector<HTMLElement>('.interview-surface')!;
      const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
      try {
        if (!reduce) {
          animation = surface.animate([{ transform: 'perspective(1000px) rotateY(0)', opacity: 1 }, { transform: 'perspective(1000px) rotateY(90deg)', opacity: 0 }], { duration: 160, easing: 'ease-in', fill: 'forwards' });
          await animation.finished;
        }
        if (generation !== transitionGeneration) return;
        revealed = true; animation?.cancel(); update();
        // The persistent reveal control retains keyboard focus during the flip.
        if (!reduce) {
          animation = surface.animate([{ transform: 'perspective(1000px) rotateY(-90deg)', opacity: 0 }, { transform: 'perspective(1000px) rotateY(0)', opacity: 1 }], { duration: 160, easing: 'ease-out' });
          await animation.finished;
        }
        announce('Answer revealed. Detailed explanation is available below the card.');
      } catch { /* Filter/mode changes cancel an in-flight transition. */ }
      finally { if (generation === transitionGeneration) { flipping = false; button.disabled = false; animation = undefined; } }
    });
    bank.querySelectorAll<HTMLButtonElement>('[data-rate]').forEach((button) => listen(button, 'click', () => {
      if (!revealed || !current) return;
      progress.ratings[current] = button.dataset.rate as Rating; save(); update(); announce(`Marked ${button.textContent}.`);
    }));
    const navigate = (direction: number) => {
      const ids = queue(); const position = ids.indexOf(current);
      resetCard();
      if (direction === 1 && position === ids.length - 1) {
        finished = true; update(); one('[data-finish] h2').focus(); return;
      }
      current = ids[Math.max(0, Math.min(ids.length - 1, position + direction))];
      closeDetails(); update(true); announce(`Question ${ids.indexOf(current) + 1} of ${ids.length}.`);
      one('[data-practice-header]').scrollIntoView({ block: 'start', behavior: 'instant' });
    };
    listen(one('[data-previous]'), 'click', () => navigate(-1)); listen(one('[data-next]'), 'click', () => navigate(1));
    listen(one('[data-review-again]'), 'click', () => { reviewQueue = queue().filter((id) => progress.ratings[id] === 'again'); resetCard(); current = queue()[0] ?? ''; update(true); });
    listen(one('[data-restart]'), 'click', () => { reviewQueue = null; resetCard(); current = queue()[0] ?? ''; update(true); });
    listen(one('[data-reset]'), 'click', () => { one('[data-reset-confirm]').hidden = false; one('[data-reset-yes]').focus(); });
    listen(one('[data-reset-no]'), 'click', () => { one('[data-reset-confirm]').hidden = true; one('[data-reset]').focus(); });
    listen(one('[data-reset-yes]'), 'click', () => {
      progress = emptyProgress(); reviewQueue = null; resetCard(); current = queue()[0] ?? ''; closeDetails(); save();
      one('[data-reset-confirm]').hidden = true; update(); one('[data-reset]').focus(); announce('Saved ratings cleared.');
    });
    const openHash = () => {
      let hash = ''; try { hash = decodeURIComponent(location.hash.slice(1)); } catch { return; }
      const target = document.getElementById(hash);
      if (!target || !bank.contains(target)) return;
      resetCard(); mode = 'read'; topic.value = ''; search.value = ''; reviewQueue = null;
      const card = target.closest<HTMLElement>('[data-question]');
      if (card?.dataset.question) current = card.dataset.question;
      update(); requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
    };
    document.querySelectorAll<HTMLAnchorElement>('.interview-rail a, .interview-mobile-topics a').forEach((link) => listen(link, 'click', (event) => {
      if (mode !== 'practice') {
        event.preventDefault();
        if (location.hash !== link.hash) history.pushState(null, '', link.hash);
        openHash(); return;
      }
      event.preventDefault();
      topic.value = link.hash.replace('#topic-', '');
      resetCard(); reviewQueue = null; update(true);
      one('[data-controls]').scrollIntoView({ block: 'start' });
    }));
    listen(bank, 'click', (event) => {
      const link = (event.target as Element).closest<HTMLAnchorElement>('a[href^="#"]');
      if (!link || !document.getElementById(link.hash.slice(1))) return;
      event.preventDefault();
      if (location.hash !== link.hash) history.pushState(null, '', link.hash);
      openHash();
    });
    listen(window, 'hashchange', openHash);
    // Expand native details for browsers that suppress closed content during printing.
    let printState: { open: boolean[] } | null = null;
    listen(window, 'beforeprint', () => {
      cancelFlip();
      const details = Array.from(bank.querySelectorAll<HTMLDetailsElement>('details'));
      printState = { open: details.map((detail) => detail.open) };
      details.forEach((detail) => { detail.open = true; }); bank.dataset.printing = 'true';
    });
    listen(window, 'afterprint', () => {
      if (!printState) return;
      bank.querySelectorAll<HTMLDetailsElement>('details').forEach((detail, i) => { detail.open = printState!.open[i]; });
      delete bank.dataset.printing; printState = null;
    });
    listen(document, 'astro:before-swap', () => { cancelFlip(); cancelAnimationFrame(disclosureFrame); abort.abort(); });
    one('[data-controls]').hidden = false; update(); openHash();
    if (!stored) one('[data-storage-note]').textContent = 'Browser storage is unavailable. Progress lasts for this visit.';
  });
}
