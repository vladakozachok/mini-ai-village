from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

DetailLevel = Literal["compact", "full"]

_CHESS_NOTATION_RE = re.compile(
    r"^(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|[a-h](?:x[a-h][1-8]|[1-8])(?:=[QRBN])?[+#]?)$"
)


OBSERVATION_SCRIPT = r"""
(args) => {
  const detail = String(args?.detail ?? "compact");
  const isCompact = detail !== "full";

  const maxElements        = Number(args?.maxElements        ?? (isCompact ? 50  : 110));
  const maxPromotedAnchors = Number(args?.maxPromotedAnchors ?? (isCompact ? 20  : 48));
  const maxBodyChars       = Number(args?.maxBodyChars       ?? (isCompact ? 360 : 1800));

  const enableBoardAdapter = args?.enableBoardAdapter !== false;
  const maxTargetSurfaces  = Number(args?.maxTargetSurfaces  ?? (isCompact ? 2  : 6));
  const maxSurfaceAnchorsDefault = Number(args?.maxSurfaceAnchors ?? (isCompact ? 16 : 32));

  // ── Helpers ─────────────────────────────────────────────────────────────

  function normalizeClassName(value) {
    if (!value) return '';
    if (typeof value === 'string') return value;
    if (typeof value.baseVal === 'string') return value.baseVal;
    return String(value);
  }

  function classTokens(el, maxItems = 6) {
    const raw = normalizeClassName(el?.className || '');
    if (!raw) return [];
    return raw.split(/\s+/).filter(Boolean).slice(0, maxItems);
  }

  function parseTransformToPixels(styleValue, el) {
    if (!styleValue) return null;

    // translate(Xpx, Ypx)  |  translate(X%, Y%)  |  translate3d(…)
    let m = styleValue.match(/translate(?:3d)?\(\s*([-\d.]+)(px|%)\s*,\s*([-\d.]+)(px|%)/i);
    if (m) {
      const w = el ? el.offsetWidth  || 0 : 0;
      const h = el ? el.offsetHeight || 0 : 0;
      const x = m[2] === '%' ? (Number(m[1]) / 100) * w : Number(m[1]);
      const y = m[4] === '%' ? (Number(m[3]) / 100) * h : Number(m[3]);
      return { x: Math.round(x), y: Math.round(y) };
    }

    // translateX(…) and/or translateY(…) as separate functions
    let tx = 0, ty = 0, found = false;
    const txm = styleValue.match(/translateX\(\s*([-\d.]+)(px|%)/i);
    const tym = styleValue.match(/translateY\(\s*([-\d.]+)(px|%)/i);
    if (txm) {
      const w = el ? el.offsetWidth || 0 : 0;
      tx = txm[2] === '%' ? (Number(txm[1]) / 100) * w : Number(txm[1]);
      found = true;
    }
    if (tym) {
      const h = el ? el.offsetHeight || 0 : 0;
      ty = tym[2] === '%' ? (Number(tym[1]) / 100) * h : Number(tym[1]);
      found = true;
    }
    if (found) return { x: Math.round(tx), y: Math.round(ty) };

    // matrix(a,b,c,d,tx,ty)
    m = styleValue.match(/matrix\(\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*[-\d.]+\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)/i);
    if (m) return { x: Number(m[1]), y: Number(m[2]) };

    // matrix3d — tx/ty are indices 12 and 13
    m = styleValue.match(/matrix3d\(([^)]+)\)/i);
    if (m) {
      const vals = m[1].split(',').map(Number);
      if (vals.length >= 14) return { x: Math.round(vals[12]), y: Math.round(vals[13]) };
    }

    return null;
  }

  function getTransformTranslate(el) {
    const inlineStyle = el.getAttribute('style') || '';
    const fromInline  = parseTransformToPixels(inlineStyle, el);
    if (fromInline) return fromInline;

    // Fall back to computed style (covers CSS-class-driven transforms)
    const computed = window.getComputedStyle(el).transform || '';
    if (computed && computed !== 'none') return parseTransformToPixels(computed, el);

    return null;
  }

  function normalizedRect(rect) {
    const vw = Math.max(1, window.innerWidth  || 1);
    const vh = Math.max(1, window.innerHeight || 1);
    return {
      x:        Math.round((rect.x / vw) * 10000) / 10000,
      y:        Math.round((rect.y / vh) * 10000) / 10000,
      width:    Math.round((rect.width  / vw) * 10000) / 10000,
      height:   Math.round((rect.height / vh) * 10000) / 10000,
      center_x: Math.round(((rect.x + rect.width  / 2) / vw) * 10000) / 10000,
      center_y: Math.round(((rect.y + rect.height / 2) / vh) * 10000) / 10000,
    };
  }

  function cssPath(el) {
    if (!el || el.nodeType !== 1) return null;
    if (el.id) return `#${CSS.escape(el.id)}`;
    const stableAttrs = ['data-testid', 'data-test', 'data-qa', 'data-cy', 'data-id'];
    for (const a of stableAttrs) {
      const v = el.getAttribute(a);
      if (v) return `[${a}="${v}"]`;
    }
    if (el.getAttribute('name'))
      return `${el.tagName.toLowerCase()}[name="${el.getAttribute('name')}"]`;
    const parts = [];
    let node = el;
    while (node && node.nodeType === 1 && parts.length < 4) {
      let part = node.tagName.toLowerCase();
      const rawCls = normalizeClassName(node.className || '');
      if (rawCls && typeof rawCls === 'string') {
        const cls = rawCls.split(/\s+/).filter(Boolean)[0];
        if (cls) part += `.${cls.replace(/[^a-zA-Z0-9_-]/g, '')}`;
      }
      const parent = node.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(s => s.tagName === node.tagName);
        if (siblings.length > 1) part += `:nth-of-type(${siblings.indexOf(node) + 1})`;
      }
      parts.unshift(part);
      node = parent;
    }
    return parts.join(' > ');
  }

  function getLabelText(el) {
    const ariaLabel = el.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel.trim();
    const ariaLabelledBy = el.getAttribute('aria-labelledby');
    if (ariaLabelledBy) {
      const labelEl = document.getElementById(ariaLabelledBy);
      if (labelEl) return (labelEl.textContent || labelEl.innerText || '').trim();
    }
    if (el.labels && el.labels.length > 0)
      return (el.labels[0].textContent || el.labels[0].innerText || '').trim();
    return (el.getAttribute('placeholder') || el.getAttribute('title') ||
            el.getAttribute('alt') || '').trim();
  }

  function getRichLabel(el) {
    // Standard ARIA / form labels first
    const base = getLabelText(el);
    if (base) return base;

    // Harvest semantically-named data attributes
    const semanticData = [
      'data-label', 'data-name', 'data-value', 'data-key',
      'data-piece', 'data-square', 'data-cell', 'data-slot',
      'data-row', 'data-col', 'data-column', 'data-index',
      'data-state', 'data-status', 'data-type', 'data-kind',
    ];
    const parts = [];
    for (const attr of semanticData) {
      const v = el.getAttribute(attr);
      if (v) parts.push(`${attr.replace('data-', '')}:${v}`);
    }
    if (parts.length) return parts.join(' ');

    return '';
  }

  function isVisible(el) {
    const rect = el.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none';
  }

  function overlapsViewport(rect) {
    return rect.bottom > 0 && rect.right > 0 &&
           rect.top < window.innerHeight && rect.left < window.innerWidth;
  }

  function areaRatio(rect) {
    const vw = Math.max(1, window.innerWidth  || 1);
    const vh = Math.max(1, window.innerHeight || 1);
    return (Math.max(0, rect.width) * Math.max(0, rect.height)) / (vw * vh);
  }

  function centerCloseness(rect) {
    const cx = rect.x + (rect.width / 2);
    const cy = rect.y + (rect.height / 2);
    const dx = Math.abs(cx - (window.innerWidth / 2)) / Math.max(1, window.innerWidth / 2);
    const dy = Math.abs(cy - (window.innerHeight / 2)) / Math.max(1, window.innerHeight / 2);
    const distance = Math.sqrt((dx * dx) + (dy * dy));
    return Math.max(0, 1 - (distance / Math.sqrt(2)));
  }

  function rectContainsCenter(outerRect, innerRect) {
    const cx = innerRect.x + (innerRect.width / 2);
    const cy = innerRect.y + (innerRect.height / 2);
    return (
      cx >= outerRect.left && cx <= outerRect.right &&
      cy >= outerRect.top  && cy <= outerRect.bottom
    );
  }

  function genericStateSignalScore(el) {
    const classText = classTokens(el, 10).join(' ').toLowerCase();
    const attrText = [
      el.getAttribute('aria-selected'),
      el.getAttribute('aria-current'),
      el.getAttribute('aria-expanded'),
      el.getAttribute('data-state'),
      el.getAttribute('data-status'),
      el.getAttribute('data-mode'),
    ].filter(Boolean).join(' ').toLowerCase();
    const combined = `${classText} ${attrText}`;
    let score = 0;
    if (/(selected|active|current|open|expanded|checked|pressed|focused|focus)/.test(combined)) score += 2;
    if (/(target|highlight|marker|ready|playing|turn)/.test(combined)) score += 1;
    return score;
  }

  function cleanText(value) {
    return String(value || '').replace(/\s+/g, ' ').trim();
  }

  function pushUniqueText(bucket, value, maxLen = 180) {
    const t = cleanText(value).slice(0, maxLen);
    if (!t || bucket.includes(t)) return;
    bucket.push(t);
  }

  function hasPointerCursor(el) {
    return window.getComputedStyle(el).cursor === 'pointer';
  }

  function isDraggable(el) {
    return el.getAttribute('draggable') === 'true';
  }

  // ── 1) Generic interactive capture ──────────────────────────────────────

  const interactiveSelector = [
    'a[href]', 'button', 'input', 'textarea', 'select',
    '[role="button"]', '[role="link"]', '[tabindex]:not([tabindex="-1"])',
  ].join(',');

  const visualInteractiveSelector = [
    '[draggable="true"]',
    '[onclick]',
    '[data-action]',
    '[data-click]',
    '[data-href]',
  ].join(',');

  const interactiveNodes = Array.from(
    document.querySelectorAll(interactiveSelector)
  ).filter(isVisible);

  const visualNodes = Array.from(
    document.querySelectorAll(visualInteractiveSelector)
  ).filter(el => isVisible(el) && !el.matches(interactiveSelector));

  const pointerNodes = Array.from(
    document.querySelectorAll('div, span, li, td, svg, g, [class]')
  ).filter(el => {
    if (!isVisible(el)) return false;
    if (el.matches(interactiveSelector) || el.matches(visualInteractiveSelector)) return false;
    return hasPointerCursor(el);
  }).slice(0, 80);

  // ── 2) Board-like UI detection ───────────────────────────────────────────

  function getBoardAdapterResult() {
    if (!enableBoardAdapter) return { enabled: false, reason: 'disabled' };

    const stamp = `${document.title}::${document.body?.children?.length ?? 0}`;
    const cache = window.__villageObsCache__;
    if (cache && cache.stamp === stamp) return cache.boardAdapter;

    const rootText = cleanText((document.body?.textContent || '').slice(0, 12000)).toLowerCase();
    const hasCanvasOrSvg = !!document.querySelector('canvas, svg');
    const hasAppRole     = !!document.querySelector('[role="application"]');
    const nameSignals    = !!document.querySelector(
      '[aria-label*="board" i], [aria-label*="grid" i], [aria-label*="map" i], [aria-label*="canvas" i], ' +
      '[data-testid*="board" i], [data-testid*="grid" i], [id*="board" i], [id*="grid" i], ' +
      '[class*="board" i], [class*="grid" i], [class*="canvas" i]'
    );
    const draggableCount  = document.querySelectorAll('[draggable="true"]').length;
    const dataGridAttrs   = document.querySelectorAll('[data-square],[data-cell],[data-slot],[data-row],[data-col]').length;
    const transformCount  = document.querySelectorAll('[style*="translate("], [style*="matrix("]').length;
    const statusHint      = /(?:turn|moves?|score|level|round|player|your move|vs)\b/i.test(rootText);

    const score =
      (hasCanvasOrSvg   ? 3 : 0) +
      (hasAppRole       ? 2 : 0) +
      (nameSignals      ? 2 : 0) +
      (transformCount  >= 20 ? 2 : transformCount >= 8 ? 1 : 0) +
      (draggableCount  >= 8  ? 2 : draggableCount >= 2 ? 1 : 0) +
      (dataGridAttrs   >= 8  ? 2 : dataGridAttrs  >= 2 ? 1 : 0) +
      (statusHint       ? 1 : 0);

    const result = {
      enabled: score >= 4,
      score,
      signals: { hasCanvasOrSvg, hasAppRole, nameSignals, transformCount,
                 draggableCount, dataGridAttrs, statusHint }
    };

    window.__villageObsCache__ = { stamp, boardAdapter: result };
    return result;
  }

  const boardAdapter = getBoardAdapterResult();

  // ── 3) Surface capture ───────────────────────────────────────────────────

  const buildSurfaces = boardAdapter.enabled && !isCompact;
  let targetSurfaces = [];
  let selectedSurfaceRects = [];

  if (boardAdapter.enabled) {
    const explicitSurfaceSelector = [
      'canvas', 'svg', '[role="application"]',
      '[aria-label*="board" i]', '[aria-label*="grid" i]',
      '[class*="board" i]',      '[class*="grid" i]',
      '[data-testid*="board" i]','[data-testid*="grid" i]',
      '[id*="board" i]',         '[id*="grid" i]',
    ].join(',');

    const containerCandidates = Array.from(
      document.querySelectorAll('main, section, article, div')
    ).slice(0, 80);

    function widgetSignalScore(el) {
      let s = 0;
      if (el.querySelector('canvas, svg')) s += 3;
      if (el.getAttribute('role') === 'application') s += 3;
      if (el.querySelector('[draggable="true"]'))   s += 2;
      if (el.querySelector('[data-square],[data-cell],[data-slot]')) s += 2;
      const clueSelector = [
        '[role]','[aria-label]','[aria-roledescription]',
        '[data-testid]','[data-test]','[data-qa]','[data-cy]','[data-id]','[data-key]',
        '[tabindex]:not([tabindex="-1"])',
        'button','a[href]','input','select','textarea',
        '[style*="translate("]','[style*="matrix("]',
        '[draggable="true"]',
      ].join(',');
      const clues = el.querySelectorAll(clueSelector).length;
      s += clues >= 25 ? 3 : clues >= 12 ? 2 : clues >= 6 ? 1 : 0;
      return s;
    }

    const explicit  = Array.from(document.querySelectorAll(explicitSurfaceSelector)).filter(isVisible);
    const heuristic = containerCandidates
      .filter(el => {
        if (!isVisible(el)) return false;
        const rect = el.getBoundingClientRect();
        if (!overlapsViewport(rect)) return false;
        const minArea = isCompact ? 0.16 : 0.10;
        return areaRatio(rect) >= minArea;
      })
      .map(el => {
        const rect = el.getBoundingClientRect();
        return { el, score: widgetSignalScore(el) + Math.min(3, Math.floor(areaRatio(rect) / 0.1)) };
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, maxTargetSurfaces)
      .map(x => x.el);

    const surfaceSet = new Set();
    const surfaces   = [];
    function addSurface(el) {
      if (!el || surfaceSet.has(el)) return;
      surfaceSet.add(el); surfaces.push(el);
    }
    explicit.forEach(addSurface);
    heuristic.forEach(addSurface);

    const childSelector = [
      '[role]','[aria-label]','[aria-roledescription]',
      '[data-testid]','[data-test]','[data-qa]','[data-cy]','[data-id]','[data-key]',
      '[data-square]','[data-cell]','[data-slot]','[data-row]','[data-col]',
      '[data-piece]','[data-state]','[data-value]','[data-index]','[data-type]',
      '[tabindex]:not([tabindex="-1"])',
      '[style*="translate("]','[style*="matrix("]',
      '[draggable="true"]',
      '[onclick]','[data-action]',
      'button','a[href]','input','select','textarea','svg','canvas',
    ].join(',');

    const selectedSurfaces = surfaces.slice(0, maxTargetSurfaces);
    selectedSurfaceRects = selectedSurfaces.map(el => el.getBoundingClientRect());

    function surfaceMembershipScore(rect) {
      let best = 0;
      for (const surfaceRect of selectedSurfaceRects) {
        if (!rectContainsCenter(surfaceRect, rect)) continue;
        best = Math.max(best, 2 + Math.min(4, areaRatio(surfaceRect) * 8));
      }
      return best;
    }

    function nodePriorityScore(el) {
      const rect = el.getBoundingClientRect();
      const tag = el.tagName.toLowerCase();
      const role = el.getAttribute('role') || '';
      const interactiveLike =
        el.matches(interactiveSelector) ||
        el.matches(visualInteractiveSelector) ||
        hasPointerCursor(el) ||
        !!role;

      let score = 0;
      score += interactiveLike ? 3 : 0;
      score += isDraggable(el) ? 2 : 0;
      score += hasPointerCursor(el) ? 1 : 0;
      score += genericStateSignalScore(el);
      score += Math.min(6, areaRatio(rect) * 24);
      score += centerCloseness(rect) * 2;
      score += surfaceMembershipScore(rect);
      if (tag === 'canvas' || tag === 'svg') score += 2;
      if (getRichLabel(el)) score += 1;
      return score;
    }

    function childPriorityScore(child) {
      const rect = child.getBoundingClientRect();
      let score = nodePriorityScore(child);
      if (getTransformTranslate(child)) score += 1.5;
      score += centerCloseness(rect);
      return score;
    }

    targetSurfaces = selectedSurfaces.map((el, index) => {
      const rect = el.getBoundingClientRect();

      const surfaceScore     = widgetSignalScore(el);
      const scaledAnchorCap = surfaceScore >= 8
        ? Math.min(maxSurfaceAnchorsDefault * 3, 96)
        : surfaceScore >= 5
          ? Math.min(maxSurfaceAnchorsDefault * 2, 64)
          : maxSurfaceAnchorsDefault;

      const selectorHits  = Array.from(el.querySelectorAll(childSelector));
      const pointerHits   = Array.from(el.querySelectorAll('div, span, li, td, g'))
        .filter(child => hasPointerCursor(child) && !selectorHits.includes(child));

      const allChildren   = [...selectorHits, ...pointerHits].filter(isVisible);

      const seenRects = new Set();
      const dedupedChildren = allChildren.filter(child => {
        const r   = child.getBoundingClientRect();
        const key = `${Math.round(r.x)},${Math.round(r.y)},${Math.round(r.width)},${Math.round(r.height)}`;
        if (seenRects.has(key)) return false;
        seenRects.add(key);
        return true;
      });

      const rankedChildren = dedupedChildren
        .map(child => ({ child, score: childPriorityScore(child) }))
        .sort((a, b) => b.score - a.score)
        .map(item => item.child);

      if (isCompact) {
        return {
          surface_id: `surface_${index}`,
          tag:        el.tagName.toLowerCase(),
          selector:   cssPath(el),
          normalized_bbox: normalizedRect(rect),
          child_anchors: rankedChildren.slice(0, scaledAnchorCap).map((child, ci) => ({
            anchor_id:   `surface_${index}_anchor_${ci}`,
            tag:         child.tagName.toLowerCase(),
            text:        cleanText(child.textContent || child.innerText || '').slice(0, 40),
            label:       getRichLabel(child).slice(0, 60),
            role:        child.getAttribute('role') || '',
            selector:    cssPath(child),
            normalized_bbox:    normalizedRect(child.getBoundingClientRect()),
            transform_translate: getTransformTranslate(child),
            draggable:   isDraggable(child) || undefined,
          })),
        };
      }

      const childTagCounts = {};
      for (const child of allChildren) {
        const key = child.tagName.toLowerCase();
        childTagCounts[key] = (childTagCounts[key] || 0) + 1;
      }

      const childAnchors = rankedChildren.slice(0, scaledAnchorCap).map((child, ci) => ({
        anchor_id:   `surface_${index}_anchor_${ci}`,
        tag:         child.tagName.toLowerCase(),
        text:        cleanText(child.textContent || child.innerText || '').slice(0, 40),
        label:       getRichLabel(child).slice(0, 60),
        role:        child.getAttribute('role') || '',
        selector:    cssPath(child),
        normalized_bbox:    normalizedRect(child.getBoundingClientRect()),
        transform_translate: getTransformTranslate(child),
        draggable:   isDraggable(child) || undefined,
        classes:     classTokens(child, 4),
      }));

      return {
        surface_id:  `surface_${index}`,
        tag:         el.tagName.toLowerCase(),
        role:        el.getAttribute('role') || '',
        selector:    cssPath(el),
        classes:     classTokens(el, 12),
        normalized_bbox: normalizedRect(rect),
        child_counts: childTagCounts,
        child_anchors: childAnchors,
      };
    });
  }

  // ── 4) Base interactive elements (now includes visual nodes) ─────────────

  const nodes   = [];
  const seenSet = new Set();
  function addNode(node) {
    if (!node || seenSet.has(node)) return;
    seenSet.add(node); nodes.push(node);
  }
  interactiveNodes.forEach(addNode);
  visualNodes.forEach(addNode);
  pointerNodes.forEach(addNode);

  const visible = nodes.filter(isVisible);
  const visibleRanked = visible
    .map(el => {
      const rect = el.getBoundingClientRect();
      const tag = el.tagName.toLowerCase();
      const role = el.getAttribute('role') || '';
      const label = getRichLabel(el);
      const text = cleanText(el.textContent || el.innerText || '');
      const interactiveLike = el.matches(interactiveSelector) ||
                              el.matches(visualInteractiveSelector) ||
                              hasPointerCursor(el) ||
                              !!role;
      let score = 0;
      score += interactiveLike ? 3 : 0;
      score += isDraggable(el) ? 2 : 0;
      score += hasPointerCursor(el) ? 1 : 0;
      score += genericStateSignalScore(el);
      score += Math.min(6, areaRatio(rect) * 24);
      score += centerCloseness(rect) * 2;
      if (tag === 'canvas' || tag === 'svg') score += 2;
      if (label) score += 1;
      if (text && text.length <= 80) score += 0.5;
      for (const surfaceRect of selectedSurfaceRects) {
        if (!rectContainsCenter(surfaceRect, rect)) continue;
        score += 2 + Math.min(4, areaRatio(surfaceRect) * 8);
      }
      return { el, score };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, maxElements * 2);

  const baseElements = visibleRanked.map(({ el, score }, index) => {
    const rect = el.getBoundingClientRect();
    const tag  = el.tagName.toLowerCase();
    const isInteractive = el.matches(interactiveSelector) ||
                          el.matches(visualInteractiveSelector) ||
                          hasPointerCursor(el);

    let value = '';
    if (tag === 'input' || tag === 'textarea') value = el.value || '';
    else if (tag === 'select') value = el.options?.[el.selectedIndex]?.text || '';

    const textRaw  = cleanText(el.textContent || el.innerText || '');
    const richLabel = getRichLabel(el);

    return {
      _score: score,
      elementId: `elem_${index}`,
      tag,
      kind:     isInteractive ? 'interactive' : 'container',
      role:     el.getAttribute('role') || '',
      text:     textRaw.slice(0, 120),
      label:    richLabel.slice(0, 120),
      value:    cleanText(value).slice(0, 120),
      href:     el.href || '',
      selector: cssPath(el),
      disabled:    !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      required:    el.hasAttribute('required'),
      checked:     !!el.checked,
      hasError:    el.classList.contains('error') || el.classList.contains('invalid') ||
                   el.getAttribute('aria-invalid') === 'true',
      isInViewport: overlapsViewport(rect),
      normalized_bbox: normalizedRect(rect),
      draggable: isDraggable(el) || undefined,
      ...(detail === "full" ? {
        classes:    classTokens(el, 8),
        type:       el.getAttribute('type') || '',
        placeholder: el.getAttribute('placeholder') || '',
        parentRole: el.closest('[role]')?.getAttribute('role') || '',
        formName:   el.closest('form')?.getAttribute('name') || '',
        x:          Math.round(rect.x),
        y:          Math.round(rect.y),
        width:      Math.round(rect.width),
        height:     Math.round(rect.height),
      } : {}),
    };
  });

  // ── 5) Promoted anchors ─────────────────────────────────────────────────

  let dedupPromoted = [];
  if (boardAdapter.enabled && targetSurfaces.length) {
    const interactiveLikeTags = new Set(['button', 'a', 'input', 'select', 'textarea']);

    const promotedAnchors = targetSurfaces
      .flatMap((surface) => (surface.child_anchors || []).map(anchor => ({ surface, anchor })))
      .map(({ surface, anchor }, idx) => {
        const bb = anchor.normalized_bbox || {};
        const area = Math.max(0, Number(bb.width || 0)) * Math.max(0, Number(bb.height || 0));
        const hasRole          = !!anchor.role;
        const hasTransform     = !!anchor.transform_translate;
        const isDrag           = !!anchor.draggable;
        const isInteractiveLike = interactiveLikeTags.has(anchor.tag) || hasRole;
        const score =
          (isInteractiveLike ? 2 : 0) +
          (hasTransform ? 1.5 : 0) +
          (isDrag ? 2 : 0) +
          Math.min(4, area * 20);

        return {
          _score:   score,
          elementId: `anchor_${idx}`,
          tag:      anchor.tag,
          kind:     'interactive',
          role:     anchor.role   || '',
          text:     anchor.text   || '',
          label:    anchor.label  || '',
          selector: anchor.selector || '',
          disabled: false,
          hasError: false,
          isInViewport: true,
          normalized_bbox: anchor.normalized_bbox,
          draggable: anchor.draggable,
        };
      })
      .filter(item => item.selector)
      .sort((a, b) => b._score - a._score)
      .slice(0, maxPromotedAnchors)
      .map(({ _score, ...rest }) => rest);

    const seenPromoted = new Set();
    dedupPromoted = promotedAnchors.filter(item => {
      const bb  = item.normalized_bbox || {};
      const key = `${item.selector}::${item.tag}:${bb.x}:${bb.y}:${bb.width}:${bb.height}`;
      if (seenPromoted.has(key)) return false;
      seenPromoted.add(key);
      return true;
    });
  }

  const combinedCandidates = [...baseElements, ...dedupPromoted]
    .sort((a, b) => (b._score || 0) - (a._score || 0));

  const seenCombined = new Set();
  const elements = combinedCandidates
    .filter(item => {
      const bb = item.normalized_bbox || {};
      const key = `${item.selector || ''}::${item.tag || ''}:${bb.x}:${bb.y}:${bb.width}:${bb.height}`;
      if (seenCombined.has(key)) return false;
      seenCombined.add(key);
      return true;
    })
    .slice(0, maxElements)
    .map(({ _score, ...rest }) => rest);

  // ── 6) Focused / salient text ────────────────────────────────────────────

  const statusCandidates = Array.from(document.querySelectorAll(
    '[role="status"], [role="alert"], [aria-live], .toast, .notification, ' +
    '.error, .warning, .success, .status, .message, .turn, .moves, .score, .player'
  ));

  const focusedSnippets = [];
  statusCandidates
    .map(n => cleanText(n.textContent || n.innerText))
    .filter(Boolean)
    .slice(0, 8)
    .forEach(t => pushUniqueText(focusedSnippets, t, 220));

  const focusedText = focusedSnippets.join(' | ').slice(0, maxBodyChars);

  function extractSalientText(maxChars) {
    const snippets = [];

    Array.from(document.querySelectorAll('h1, h2, h3, [data-testid*="title" i], [class*="title" i]'))
      .slice(0, 12)
      .forEach(n => pushUniqueText(snippets, n.textContent || n.innerText, 180));

    Array.from(document.querySelectorAll('button, a[href], [role="button"], [role="link"]'))
      .slice(0, 40)
      .forEach(n => {
        const label = n.getAttribute('aria-label') || (n.textContent || n.innerText || '');
        pushUniqueText(snippets, label, 100);
      });

    Array.from(document.querySelectorAll('input, textarea'))
      .slice(0, 20)
      .forEach(n => {
        const v = String(n.value || '');
        if (/https?:\/\//i.test(v)) pushUniqueText(snippets, v, 220);
      });

    const filtered = snippets.filter(s => !focusedText.includes(s));
    const joined   = filtered.join(' | ').slice(0, maxChars);
    if (joined) return joined;

    return cleanText(document.body?.textContent || document.body?.innerText || '')
      .slice(0, maxChars);
  }

  const clippedBody = extractSalientText(maxBodyChars);

  // ── 7) Page state ─────────────────────────────────────────────────────────

  const pageState = {
    hasModal:   !!document.querySelector('[role="dialog"], .modal, [aria-modal="true"]'),
    hasLoading: !!document.querySelector('[aria-busy="true"], .loading, .spinner'),
    alertCount: document.querySelectorAll('[role="alert"]').length,
    hasInputs:  interactiveNodes.some(el =>
      el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT'
    ),
    hasClickables: [...interactiveNodes, ...visualNodes, ...pointerNodes].some(el =>
      el.tagName === 'BUTTON' || el.tagName === 'A' ||
      el.getAttribute('role') === 'button' || el.getAttribute('role') === 'link' ||
      hasPointerCursor(el) || isDraggable(el)
    ),
  };

  return {
    detail,
    title:        document.title || '',
    focusedText,
    bodyText:     clippedBody,
    elements,
    pageState,
    adapters:      { board_like: boardAdapter },
    targetSurfaces,
  };
}
"""


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _one_line(text: str) -> str:
    return " ".join((text or "").replace("\r", " ").replace("\n", " ").split())


def _prune_element(el: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    pruned: dict[str, Any] = {
        "index": int(el.get("index", -1)),
        "elementId": el.get("elementId", ""),
        "tag": el.get("tag", ""),
        "kind": el.get("kind", ""),
        "role": el.get("role") or "",
        "text": _truncate(_one_line(el.get("text", "")), 120),
        "label": _truncate(_one_line(el.get("label", "")), 120),
        "value": _truncate(_one_line(el.get("value", "")), 120),
        "href": _truncate(el.get("href", ""), 180),
        "selector": el.get("selector", ""),
        "disabled": bool(el.get("disabled")),
        "required": bool(el.get("required")),
        "checked": bool(el.get("checked")),
        "hasError": bool(el.get("hasError")),
        "isInViewport": bool(el.get("isInViewport", True)),
        "normalized_bbox": el.get("normalized_bbox", {}),
    }
    if el.get("draggable"):
        pruned["draggable"] = True
    for key in ("surface_id", "surface_order", "surface_rel_x", "surface_rel_y"):
        if key in el:
            pruned[key] = el[key]
    for key in (
        "game_kind",
        "game_role",
        "game_surface_id",
        "game_square",
        "piece_color",
        "click_selector",
        "click_rel_x",
        "click_rel_y",
    ):
        if key in el:
            pruned[key] = el[key]
    if not compact:
        for k in (
            "classes",
            "type",
            "placeholder",
            "parentRole",
            "formName",
            "x",
            "y",
            "width",
            "height",
        ):
            if k in el:
                pruned[k] = el[k]
    return pruned


def _prune_surface_anchor(anchor: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    pruned: dict[str, Any] = {
        "anchor_id": anchor.get("anchor_id", ""),
        "tag": anchor.get("tag", ""),
        "text": _truncate(_one_line(anchor.get("text", "")), 40),
        "label": _truncate(_one_line(anchor.get("label", "")), 60),
        "role": anchor.get("role") or "",
        "selector": anchor.get("selector", ""),
        "normalized_bbox": anchor.get("normalized_bbox", {}),
    }
    if anchor.get("transform_translate"):
        pruned["transform_translate"] = anchor["transform_translate"]
    if anchor.get("draggable"):
        pruned["draggable"] = True
    for key in ("surface_order", "surface_rel_x", "surface_rel_y"):
        if key in anchor:
            pruned[key] = anchor[key]
    if not compact and "classes" in anchor:
        pruned["classes"] = anchor["classes"]
    return pruned


def _prune_surface(surface: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    pruned: dict[str, Any] = {
        "surface_id": surface.get("surface_id", ""),
        "tag": surface.get("tag", ""),
        "selector": surface.get("selector", ""),
        "normalized_bbox": surface.get("normalized_bbox", {}),
    }
    if role := surface.get("role"):
        pruned["role"] = role
    anchors = surface.get("child_anchors") or []
    if anchors:
        cap = 64 if compact else 96
        pruned["child_anchors"] = [
            _prune_surface_anchor(a, compact=compact)
            for a in anchors[:cap]
            if isinstance(a, dict)
        ]
    if not compact:
        for key in ("classes", "child_counts"):
            if key in surface:
                pruned[key] = surface[key]
    return pruned


def _bbox_area(bbox: dict[str, Any]) -> float:
    try:
        return max(0.0, float(bbox.get("width", 0.0))) * max(
            0.0, float(bbox.get("height", 0.0))
        )
    except (TypeError, ValueError):
        return 0.0


def _bbox_center(bbox: dict[str, Any]) -> tuple[float, float]:
    try:
        if bbox.get("center_y") is not None and bbox.get("center_x") is not None:
            return float(bbox["center_y"]), float(bbox["center_x"])
        y = float(bbox.get("y", 0.0))
        x = float(bbox.get("x", 0.0))
        h = float(bbox.get("height", 0.0))
        w = float(bbox.get("width", 0.0))
        return y + (h / 2.0), x + (w / 2.0)
    except (TypeError, ValueError):
        return 0.0, 0.0


def _surface_relative_center(
    anchor_bbox: dict[str, Any],
    surface_bbox: dict[str, Any],
) -> tuple[float, float] | None:
    try:
        ay, ax = _bbox_center(anchor_bbox)
        sy = float(surface_bbox.get("y", 0.0))
        sx = float(surface_bbox.get("x", 0.0))
        sh = max(1e-6, float(surface_bbox.get("height", 0.0)))
        sw = max(1e-6, float(surface_bbox.get("width", 0.0)))
        rel_x = max(0.0, min(1.0, (ax - sx) / sw))
        rel_y = max(0.0, min(1.0, (ay - sy) / sh))
        return round(rel_x, 4), round(rel_y, 4)
    except (TypeError, ValueError):
        return None


def _selector_bbox_key(selector: str, bbox: dict[str, Any]) -> str:
    y, x = _bbox_center(bbox)
    return f"{selector}::{round(x, 4)}::{round(y, 4)}::{round(_bbox_area(bbox), 6)}"


def _spatial_anchor_sort_key(anchor: dict[str, Any]) -> tuple[float, float, str]:
    bbox = anchor.get("normalized_bbox", {})
    cy, cx = _bbox_center(bbox)
    return cy, cx, str(anchor.get("selector", ""))


def _is_decorative_board_highlight(anchor: dict[str, Any]) -> bool:
    selector = str(anchor.get("selector", "")).lower()
    if "last-move" in selector:
        return True
    if "premove" in selector and "move-dest" not in selector:
        return True
    return False


def _annotate_surface_anchor_positions(surface: dict[str, Any]) -> dict[str, Any]:
    anchors = surface.get("child_anchors") or []
    if not anchors:
        return surface

    enriched = dict(surface)
    surface_bbox = surface.get("normalized_bbox", {})
    ordered = []
    for idx, anchor in enumerate(sorted(anchors, key=_spatial_anchor_sort_key)):
        anchor2 = dict(anchor)
        anchor2["surface_order"] = idx
        rel = _surface_relative_center(anchor2.get("normalized_bbox", {}), surface_bbox)
        if rel is not None:
            anchor2["surface_rel_x"], anchor2["surface_rel_y"] = rel
        ordered.append(anchor2)
    enriched["child_anchors"] = ordered
    return enriched


def _board_like_surface_candidates(
    surfaces: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    anchored = [
        s for s in surfaces if isinstance(s, dict) and (s.get("child_anchors") or [])
    ]
    if not anchored:
        return []

    focused = [
        s
        for s in anchored
        if s.get("tag") not in {"body", "html"}
        and _bbox_area(s.get("normalized_bbox", {})) <= 0.7
    ]
    candidates = focused or anchored
    return sorted(
        candidates,
        key=lambda s: (
            _bbox_area(s.get("normalized_bbox", {})),
            s.get("surface_id", ""),
        ),
    )


def _reorder_board_like_elements(
    elements: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    *,
    chess_like_surface_id: str = "",
) -> list[dict[str, Any]]:
    candidates = _board_like_surface_candidates(surfaces)
    if not candidates:
        return elements

    element_by_key: dict[str, dict[str, Any]] = {}
    for element in elements:
        key = _selector_bbox_key(
            str(element.get("selector", "")),
            element.get("normalized_bbox", {}),
        )
        element_by_key.setdefault(key, element)

    ordered_board_elements: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for surface in candidates:
        for anchor in surface.get("child_anchors") or []:
            if (
                chess_like_surface_id
                and str(surface.get("surface_id", "")) == chess_like_surface_id
                and _is_decorative_board_highlight(anchor)
            ):
                continue
            key = _selector_bbox_key(
                str(anchor.get("selector", "")),
                anchor.get("normalized_bbox", {}),
            )
            if key in used_keys or key not in element_by_key:
                continue
            used_keys.add(key)
            element = dict(element_by_key[key])
            element["surface_id"] = surface.get("surface_id", "")
            element["surface_order"] = int(
                anchor.get("surface_order", len(ordered_board_elements))
            )
            if anchor.get("surface_rel_x") is not None:
                element["surface_rel_x"] = anchor["surface_rel_x"]
            if anchor.get("surface_rel_y") is not None:
                element["surface_rel_y"] = anchor["surface_rel_y"]
            ordered_board_elements.append(element)

    if not ordered_board_elements:
        return elements

    remaining = [
        dict(element)
        for element in elements
        if _selector_bbox_key(
            str(element.get("selector", "")), element.get("normalized_bbox", {})
        )
        not in used_keys
    ]
    reordered = ordered_board_elements + remaining
    for idx, element in enumerate(reordered):
        element["index"] = idx
    return reordered


def _anchor_looks_chess_like(anchor: dict[str, Any]) -> bool:
    tag = str(anchor.get("tag", "")).lower()
    selector = str(anchor.get("selector", "")).lower()
    return (
        tag in {"piece", "square"}
        or "> piece" in selector
        or "> square" in selector
        or "piece." in selector
        or "square." in selector
    )


def _pick_chess_like_surface(surfaces: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: tuple[int, float, dict[str, Any]] | None = None
    for surface in _board_like_surface_candidates(surfaces):
        anchors = [
            a
            for a in (surface.get("child_anchors") or [])
            if isinstance(a, dict)
            and _anchor_looks_chess_like(a)
            and isinstance(a.get("surface_rel_x"), (int, float))
            and isinstance(a.get("surface_rel_y"), (int, float))
        ]
        if len(anchors) < 6:
            continue

        piece_count = sum(
            1 for a in anchors if str(a.get("tag", "")).lower() == "piece"
        )
        square_count = sum(
            1 for a in anchors if str(a.get("tag", "")).lower() == "square"
        )
        unique_cols = len(
            {min(7, max(0, int(float(a["surface_rel_x"]) * 8))) for a in anchors}
        )
        unique_rows = len(
            {min(7, max(0, int(float(a["surface_rel_y"]) * 8))) for a in anchors}
        )
        color_markers = sum(
            1
            for a in anchors
            if ".white" in str(a.get("selector", "")).lower()
            or ".black" in str(a.get("selector", "")).lower()
        )
        score = (
            len(anchors)
            + (piece_count * 2)
            + square_count
            + (unique_cols * 3)
            + (unique_rows * 3)
            + color_markers
            + (10 if str(surface.get("tag", "")).lower() not in {"body", "html"} else 0)
        )
        candidate = (score, -_bbox_area(surface.get("normalized_bbox", {})), surface)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best[2] if best else None


def _infer_chess_orientation(surface: dict[str, Any]) -> str | None:
    white_y: list[float] = []
    black_y: list[float] = []
    for anchor in surface.get("child_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        rel_y = anchor.get("surface_rel_y")
        if not isinstance(rel_y, (int, float)):
            continue
        selector = str(anchor.get("selector", "")).lower()
        if ".white" in selector:
            white_y.append(float(rel_y))
        elif ".black" in selector:
            black_y.append(float(rel_y))
    if len(white_y) < 2 or len(black_y) < 2:
        return None
    return (
        "white"
        if (sum(white_y) / len(white_y)) > (sum(black_y) / len(black_y))
        else "black"
    )


def _square_name_from_relative(rel_x: float, rel_y: float, orientation: str) -> str:
    col = min(7, max(0, int(float(rel_x) * 8)))
    row = min(7, max(0, int(float(rel_y) * 8)))
    if orientation == "white":
        return f"{'abcdefgh'[col]}{8 - row}"
    return f"{'hgfedcba'[col]}{row + 1}"


def _quantized_square_center(rel_x: float, rel_y: float) -> tuple[float, float]:
    col = min(7, max(0, int(float(rel_x) * 8)))
    row = min(7, max(0, int(float(rel_y) * 8)))
    return round((col + 0.5) / 8.0, 4), round((row + 0.5) / 8.0, 4)


def _build_game_like_adapter(
    url: str, surfaces: list[dict[str, Any]]
) -> dict[str, Any]:
    surface = _pick_chess_like_surface(surfaces)
    if not surface or not str(surface.get("selector", "")).strip():
        return {"enabled": False}

    orientation = _infer_chess_orientation(surface)
    if orientation is None:
        return {"enabled": False}

    highlighted_squares: list[str] = []
    seen_squares: set[str] = set()
    white_count = 0
    black_count = 0
    for anchor in surface.get("child_anchors") or []:
        if not isinstance(anchor, dict) or not _anchor_looks_chess_like(anchor):
            continue
        rel_x = anchor.get("surface_rel_x")
        rel_y = anchor.get("surface_rel_y")
        if not isinstance(rel_x, (int, float)) or not isinstance(rel_y, (int, float)):
            continue
        square = _square_name_from_relative(float(rel_x), float(rel_y), orientation)
        selector = str(anchor.get("selector", "")).lower()
        if ".white" in selector:
            white_count += 1
        elif ".black" in selector:
            black_count += 1
        elif (
            "last-move" in selector or "move-dest" in selector or "premove" in selector
        ):
            if square not in seen_squares:
                seen_squares.add(square)
                highlighted_squares.append(square)

    site = urlparse(url).hostname or "unknown"
    return {
        "enabled": True,
        "kind": "chess_like",
        "site": site,
        "surface_id": surface.get("surface_id", ""),
        "surface_selector": surface.get("selector", ""),
        "orientation": orientation,
        "player_color": orientation,
        "highlighted_squares": highlighted_squares[:8],
        "piece_counts": {"white": white_count, "black": black_count},
    }


def _annotate_game_like_elements(
    elements: list[dict[str, Any]],
    surfaces: list[dict[str, Any]],
    game_like: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (isinstance(game_like, dict) and game_like.get("enabled")):
        return elements
    if game_like.get("kind") != "chess_like":
        return elements

    surface_id = str(game_like.get("surface_id", ""))
    click_selector = str(game_like.get("surface_selector", "")).strip()
    orientation = str(game_like.get("orientation", "")).strip()
    if not surface_id or not click_selector or orientation not in {"white", "black"}:
        return elements

    selected_surface = next(
        (surface for surface in surfaces if surface.get("surface_id") == surface_id),
        None,
    )
    if not isinstance(selected_surface, dict):
        return elements

    anchor_by_key: dict[str, dict[str, Any]] = {}
    for anchor in selected_surface.get("child_anchors") or []:
        if not isinstance(anchor, dict):
            continue
        key = _selector_bbox_key(
            str(anchor.get("selector", "")),
            anchor.get("normalized_bbox", {}),
        )
        anchor_by_key[key] = anchor

    annotated: list[dict[str, Any]] = []
    for element in elements:
        key = _selector_bbox_key(
            str(element.get("selector", "")),
            element.get("normalized_bbox", {}),
        )
        anchor = anchor_by_key.get(key)
        if not anchor:
            annotated.append(element)
            continue

        rel_x = anchor.get("surface_rel_x")
        rel_y = anchor.get("surface_rel_y")
        if not isinstance(rel_x, (int, float)) or not isinstance(rel_y, (int, float)):
            annotated.append(element)
            continue

        el2 = dict(element)
        el2["game_kind"] = "chess_like"
        el2["game_surface_id"] = surface_id
        el2["game_role"] = str(anchor.get("tag", "")).lower() or "anchor"
        el2["game_square"] = _square_name_from_relative(
            float(rel_x), float(rel_y), orientation
        )
        square_rel_x, square_rel_y = _quantized_square_center(
            float(rel_x), float(rel_y)
        )
        el2["click_selector"] = click_selector
        el2["click_rel_x"] = square_rel_x
        el2["click_rel_y"] = square_rel_y
        selector = str(anchor.get("selector", "")).lower()
        if ".white" in selector:
            el2["piece_color"] = "white"
        elif ".black" in selector:
            el2["piece_color"] = "black"
        annotated.append(el2)

    return annotated


def _is_chess_history_entry(element: dict[str, Any]) -> bool:
    tag = str(element.get("tag", "")).lower()
    selector = str(element.get("selector", "")).lower()
    if tag == "kwdb" or "kwdb" in selector:
        return True

    if element.get("game_role") in {"piece", "square"} or element.get("game_square"):
        return False

    text = _one_line(str(element.get("text", ""))).strip()
    label = _one_line(str(element.get("label", ""))).strip()
    if label or not text or len(text) > 8:
        return False
    if any(token in selector for token in ("cg-board", "piece", "square")):
        return False
    return _CHESS_NOTATION_RE.fullmatch(text) is not None


def _filter_game_like_elements(
    elements: list[dict[str, Any]],
    game_like: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (isinstance(game_like, dict) and game_like.get("enabled")):
        return elements
    if game_like.get("kind") != "chess_like":
        return elements

    filtered = [
        dict(element) for element in elements if not _is_chess_history_entry(element)
    ]
    if len(filtered) == len(elements):
        return elements

    for idx, element in enumerate(filtered):
        element["index"] = idx
    return filtered


def _chess_like_element_priority(
    element: dict[str, Any],
    *,
    player_color: str,
) -> tuple[int, int]:
    selector = str(element.get("selector", "")).lower()
    game_role = str(element.get("game_role", "")).lower()
    piece_color = str(element.get("piece_color", "")).lower()
    surface_order = int(element.get("surface_order", 9999))

    is_decorative = any(
        token in selector for token in ("last-move", "premove", "selected")
    )
    is_move_dest = "move-dest" in selector

    if game_role == "piece" and piece_color == player_color:
        return 0, surface_order
    if is_move_dest:
        return 1, surface_order
    if game_role == "square" and not is_decorative:
        return 2, surface_order
    if game_role not in {"", "piece", "square"}:
        return 3, surface_order
    if game_role == "piece" and piece_color and piece_color != player_color:
        return 4, surface_order
    if is_decorative:
        return 5, surface_order
    return 6, surface_order


def _rerank_game_like_elements(
    elements: list[dict[str, Any]],
    game_like: dict[str, Any],
) -> list[dict[str, Any]]:
    if not (isinstance(game_like, dict) and game_like.get("enabled")):
        return elements
    if game_like.get("kind") != "chess_like":
        return elements

    player_color = str(
        game_like.get("player_color", game_like.get("orientation", ""))
    ).lower()
    if player_color not in {"white", "black"}:
        return elements

    board_elements = [
        dict(element)
        for element in elements
        if element.get("game_kind") == "chess_like"
    ]
    if not board_elements:
        return elements

    board_keys = {
        _selector_bbox_key(
            str(element.get("selector", "")), element.get("normalized_bbox", {})
        )
        for element in board_elements
    }
    remaining = [
        dict(element)
        for element in elements
        if _selector_bbox_key(
            str(element.get("selector", "")), element.get("normalized_bbox", {})
        )
        not in board_keys
    ]

    ranked_board = sorted(
        board_elements,
        key=lambda element: _chess_like_element_priority(
            element,
            player_color=player_color,
        ),
    )
    reordered = ranked_board + remaining
    for idx, element in enumerate(reordered):
        element["index"] = idx
    return reordered


def _game_like_visible_text_suffix(
    elements: list[dict[str, Any]],
    game_like: dict[str, Any],
) -> str:
    if not (isinstance(game_like, dict) and game_like.get("enabled")):
        return ""
    if game_like.get("kind") != "chess_like":
        return ""

    history: list[str] = []
    for element in elements:
        if not _is_chess_history_entry(element):
            continue
        text = _one_line(str(element.get("text", ""))).strip()
        if text:
            history.append(text)

    if not history:
        return ""

    recent = history[-8:]
    return " Recent moves: " + " ".join(recent)


def _build_available_actions(
    page_state: dict[str, Any], adapters: dict[str, Any]
) -> dict[str, Any]:
    hints: list[str] = []
    if page_state.get("hasModal"):
        hints.append("modal is open — interact with modal elements or press Escape")
    if page_state.get("hasLoading"):
        hints.append("page is loading — avoid clicks until complete")
    if not page_state.get("hasInputs"):
        hints.append("no input fields visible — type/get_value unlikely to help")
    if not page_state.get("hasClickables"):
        hints.append("no clickable elements visible — consider scroll or navigate")

    result: dict[str, Any] = {
        "types": [
            "navigate",
            "click",
            "click_index",
            "click_relative",
            "scroll",
            "keypress",
            "type",
            "get_value",
        ],
        "click_target_mode": "index_or_selector",
    }
    game_like = adapters.get("game_like", {}) if isinstance(adapters, dict) else {}
    if isinstance(game_like, dict) and game_like.get("enabled"):
        hints.append(
            "game board detected — board elements may include game_square metadata; prefer click_index"
        )
    if hints:
        result["hints"] = hints
    return result


def build_observation_from_raw(
    *,
    raw: dict[str, Any],
    url: str,
    detail: DetailLevel = "compact",
) -> dict[str, Any]:
    compact = detail != "full"

    raw_surfaces = raw.get("targetSurfaces") or []
    surface_limit = 4 if compact else 8
    surfaces = [
        _annotate_surface_anchor_positions(_prune_surface(s, compact=compact))
        for s in raw_surfaces[:surface_limit]
        if isinstance(s, dict)
    ]

    raw_elements = raw.get("elements") or []
    max_elems = 50 if compact else 110
    elements: list[dict[str, Any]] = []
    for idx, el in enumerate(raw_elements[:max_elems]):
        if not isinstance(el, dict):
            continue
        el2 = dict(el)
        el2["index"] = idx
        elements.append(_prune_element(el2, compact=compact))
    adapters = (
        dict(raw.get("adapters", {}))
        if isinstance(raw.get("adapters", {}), dict)
        else {}
    )
    board_like = adapters.get("board_like", {}) if isinstance(adapters, dict) else {}
    visible_text_suffix = ""
    if isinstance(board_like, dict) and board_like.get("enabled"):
        game_like = _build_game_like_adapter(url, surfaces)
        elements = _reorder_board_like_elements(
            elements,
            surfaces,
            chess_like_surface_id=(
                str(game_like.get("surface_id", "")) if game_like.get("enabled") else ""
            ),
        )
        if game_like.get("enabled"):
            adapters["game_like"] = game_like
            elements = _annotate_game_like_elements(elements, surfaces, game_like)
            visible_text_suffix = _game_like_visible_text_suffix(elements, game_like)
            elements = _filter_game_like_elements(elements, game_like)
            elements = _rerank_game_like_elements(elements, game_like)

    FOCUSED_LIMIT = 360 if compact else 1800
    BODY_LIMIT = 360 if compact else 1800
    VISIBLE_LIMIT = 200 if compact else 700

    focused = _truncate(_one_line(raw.get("focusedText", "")), FOCUSED_LIMIT)
    body = _truncate(_one_line(raw.get("bodyText", "")), BODY_LIMIT)
    visible_text = _truncate(f"{body}{visible_text_suffix}", VISIBLE_LIMIT)

    page_state = raw.get("pageState", {})

    return {
        "url": url,
        "title": raw.get("title", ""),
        "focused_text": focused,
        "visible_text": visible_text,
        "page_state": page_state,
        "available_actions": _build_available_actions(page_state, adapters),
        "elements": elements,
        "adapters": adapters,
        "target_surfaces": surfaces,
    }
