/** Read logical source lines, never generated gutters or visual wrapping. */
export function getCodeText(code: Element): string {
  const lines = code.querySelectorAll('.line, .ec-line');
  return lines.length
    ? Array.from(lines, (line) => (line.querySelector('.code') ?? line).textContent ?? '').join('\n')
    : code.textContent ?? '';
}

export function setupCodeViews() {
  const icons = document.querySelector<HTMLTemplateElement>('#code-copy-icons');
  const updateScrollFocus = (pre: Element) => {
    const scrolls = pre.scrollWidth > pre.clientWidth + 1 || pre.scrollHeight > pre.clientHeight + 1;
    if (scrolls) pre.setAttribute('tabindex', '0');
    else pre.removeAttribute('tabindex');
  };
  const scrollObserver = new ResizeObserver((entries) => {
    for (const { target } of entries) updateScrollFocus(target);
  });
  document.querySelectorAll<HTMLElement>('pre[data-language], pre.astro-code').forEach((pre) => {
    if (pre.closest('.code-block-wrapper, .astryx-code-block')) return;
    const code = pre.querySelector('code');
    if (!code) return;
    const wrapper = document.createElement('figure');
    wrapper.className = 'code-block-wrapper';
    wrapper.dataset.wrap = 'true';
    const source = getCodeText(code).trimEnd();
    const compact = source.length <= 120 && !source.includes('\n') && !code.querySelector('.diff, .highlighted');
    wrapper.toggleAttribute('data-compact', compact);
    const toolbar = document.createElement('figcaption');
    toolbar.className = 'code-toolbar';
    const language = document.createElement('span');
    language.className = 'code-language';
    language.textContent = pre.dataset.codeTitle || pre.dataset.language || 'Code';
    language.title = language.textContent;
    language.dir = 'auto';
    pre.dir = 'ltr';
    pre.setAttribute('aria-label', `${language.textContent} code`);
    for (const line of code.querySelectorAll(':scope > .line')) {
      const text = document.createElement('span');
      text.className = 'code-line-source';
      text.append(...line.childNodes);
      line.append(text);
    }
    const controls = document.createElement('div');
    controls.className = 'code-controls';
    const button = (label: string, className: string) => {
      const control = document.createElement('button');
      control.type = 'button'; control.textContent = label; control.className = className;
      control.dataset.butterButton = 'ghost';
      controls.append(control);
      return control;
    };
    const wrap = button('Wrap', 'code-wrap-button');
    wrap.setAttribute('aria-label', 'Wrap lines');
    wrap.setAttribute('aria-pressed', 'true');
    wrap.addEventListener('click', () => {
      const enabled = wrapper.dataset.wrap !== 'true';
      wrapper.dataset.wrap = String(enabled);
      wrap.setAttribute('aria-pressed', String(enabled));
      pre.scrollLeft = 0;
      updateScrollFocus(pre);
    });
    const expand = button('Expand', 'code-expand-button');
    expand.setAttribute('aria-haspopup', 'dialog');
    if (compact) {
      wrap.hidden = true;
      expand.hidden = true;
    }
    const copy = button('Copy', 'copy-button');
    copy.setAttribute('aria-label', 'Copy code to clipboard');
    copy.title = 'Copy code';
    if (icons) {
      copy.replaceChildren(icons.content.cloneNode(true));
      copy.classList.add('code-copy-icon-button');
    }
    const status = document.createElement('span');
    status.className = 'code-status'; status.setAttribute('role', 'status');
    const setCopyFeedback = (state: 'idle' | 'copied' | 'failed') => {
      copy.dataset.copyState = state;
      copy.title = state === 'copied' ? 'Copied' : state === 'failed' ? 'Retry copying code' : 'Copy code';
      if (!icons) copy.textContent = state === 'idle' ? 'Copy' : copy.title;
    };
    let feedback: ReturnType<typeof setTimeout>;
    copy.addEventListener('click', async () => {
      clearTimeout(feedback);
      try {
        await navigator.clipboard.writeText(getCodeText(code));
        setCopyFeedback('copied'); status.textContent = 'Code copied';
      } catch {
        setCopyFeedback('failed'); status.textContent = 'Copy failed. Select the code to copy it manually.';
      }
      feedback = setTimeout(() => { setCopyFeedback('idle'); status.textContent = ''; }, 2000);
    });
    let dialog: HTMLDialogElement | undefined;
    expand.addEventListener('click', () => {
      if (dialog?.open) { dialog.close(); return; }
      const placeholder = document.createElement('div');
      placeholder.style.height = `${wrapper.getBoundingClientRect().height}px`;
      placeholder.style.marginBlock = getComputedStyle(wrapper).marginBlock;
      const tone = wrapper.closest<HTMLElement>('[data-topic-tone]')?.dataset.topicTone;
      wrapper.replaceWith(placeholder);
      dialog = document.createElement('dialog');
      dialog.className = 'code-dialog';
      dialog.setAttribute('aria-label', `${language.textContent} code — expanded view`);
      if (tone) dialog.dataset.topicTone = tone;
      expand.textContent = 'Close'; expand.removeAttribute('aria-haspopup');
      const previousOverflow = document.documentElement.style.overflow;
      dialog.append(wrapper);
      document.body.append(dialog);
      dialog.addEventListener('close', () => {
        placeholder.replaceWith(wrapper);
        expand.textContent = 'Expand'; expand.setAttribute('aria-haspopup', 'dialog');
        document.documentElement.style.overflow = previousOverflow;
        dialog?.remove(); dialog = undefined;
        expand.focus({ preventScroll: true });
      }, { once: true });
      dialog.showModal();
      document.documentElement.style.overflow = 'hidden';
      expand.focus({ preventScroll: true });
    });
    toolbar.append(language, controls, status);
    pre.before(wrapper); wrapper.append(toolbar, pre);
    scrollObserver.observe(pre);
    document.addEventListener('astro:before-swap', () => { clearTimeout(feedback); dialog?.close(); }, { once: true });
  });
  document.addEventListener('astro:before-swap', () => scrollObserver.disconnect(), { once: true });
}

export function setupTableOverflow() {
  const regions = document.querySelectorAll<HTMLElement>('.post-content .table-scroll, .post-content .rag-table');
  const observers: ResizeObserver[] = [];
  regions.forEach((region) => {
    const table = region.querySelector('table');
    if (!table || region.dataset.observed) return;
    region.dataset.observed = 'true';
    const update = () => { region.dataset.overflow = String(region.scrollWidth > region.clientWidth + 1); };
    const observer = new ResizeObserver(update);
    observer.observe(region); observer.observe(table); observers.push(observer); update();
  });
  document.addEventListener('astro:before-swap', () => observers.forEach((observer) => observer.disconnect()), { once: true });
}
