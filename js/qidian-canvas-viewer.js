(function () {
  const palette = {
    "1": { bg: "#fee2e2", border: "#b91c1c", text: "#7f1d1d", bgDark: "#3f1d22", borderDark: "#f87171", textDark: "#fecaca" },
    "2": { bg: "#ffedd5", border: "#c2410c", text: "#7c2d12", bgDark: "#3b2416", borderDark: "#fb923c", textDark: "#fed7aa" },
    "3": { bg: "#dcfce7", border: "#15803d", text: "#14532d", bgDark: "#143324", borderDark: "#4ade80", textDark: "#bbf7d0" },
    "4": { bg: "#dbeafe", border: "#1d4ed8", text: "#172554", bgDark: "#14233f", borderDark: "#60a5fa", textDark: "#bfdbfe" },
    "5": { bg: "#fef3c7", border: "#b45309", text: "#78350f", bgDark: "#3b2f12", borderDark: "#facc15", textDark: "#fde68a" },
    "6": { bg: "#f3e8ff", border: "#7e22ce", text: "#581c87", bgDark: "#2e2146", borderDark: "#c084fc", textDark: "#e9d5ff" },
  };

  const fallbackColor = {
    bg: "#f8fafc",
    border: "#475569",
    text: "#0f172a",
    bgDark: "#1e293b",
    borderDark: "#94a3b8",
    textDark: "#e2e8f0",
  };

  function colorFor(item) {
    return palette[String(item && item.color)] || fallbackColor;
  }

  function cssVars(color, prefix) {
    const name = prefix || "node";
    return [
      `--${name}-bg:${color.bg}`,
      `--${name}-border:${color.border}`,
      `--${name}-text:${color.text}`,
      `--${name}-bg-dark:${color.bgDark}`,
      `--${name}-border-dark:${color.borderDark}`,
      `--${name}-text-dark:${color.textDark}`,
    ].join(";");
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function slugifyHeading(value) {
    return String(value)
      .trim()
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, "-")
      .replace(/^-+|-+$/g, "");
  }

  function inlineMarkdown(value) {
    let html = escapeHtml(value);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/`(.+?)`/g, "<code>$1</code>");
    html = html.replace(/!\[\[([^\]]+)\]\]/g, "$1");
    html = html.replace(/\[\[([^|\]#]+)#([^|\]]+)\|([^\]]+)\]\]/g, function (_, note, heading, label) {
      return `<a href="#${slugifyHeading(heading)}">${escapeHtml(label)}</a>`;
    });
    html = html.replace(/\[\[([^|\]]+)\|([^\]]+)\]\]/g, "$2");
    html = html.replace(/\[\[([^\]]+)\]\]/g, "$1");
    return html;
  }

  function markdownToHtml(markdown) {
    const blocks = [];
    let currentList = null;

    String(markdown || "").split(/\r?\n/).forEach((line) => {
      const trimmed = line.trim();
      if (!trimmed) {
        if (currentList) {
          blocks.push(`</${currentList}>`);
          currentList = null;
        }
        return;
      }

      const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        if (currentList) {
          blocks.push(`</${currentList}>`);
          currentList = null;
        }
        const level = Math.min(heading[1].length + 2, 6);
        blocks.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
        return;
      }

      const bullet = trimmed.match(/^[-*]\s+(.+)$/);
      if (bullet) {
        if (currentList !== "ul") {
          if (currentList) blocks.push(`</${currentList}>`);
          blocks.push("<ul>");
          currentList = "ul";
        }
        blocks.push(`<li>${inlineMarkdown(bullet[1])}</li>`);
        return;
      }

      if (currentList) {
        blocks.push(`</${currentList}>`);
        currentList = null;
      }
      blocks.push(`<p>${inlineMarkdown(trimmed)}</p>`);
    });

    if (currentList) blocks.push(`</${currentList}>`);
    return blocks.join("");
  }

  function nodePoint(node, side) {
    const x = node.x;
    const y = node.y;
    const w = node.width;
    const h = node.height;
    if (side === "left") return { x, y: y + h / 2 };
    if (side === "right") return { x: x + w, y: y + h / 2 };
    if (side === "top") return { x: x + w / 2, y };
    if (side === "bottom") return { x: x + w / 2, y: y + h };
    return { x: x + w / 2, y: y + h / 2 };
  }

  function edgePath(edge, nodesById) {
    const from = nodesById.get(edge.fromNode);
    const to = nodesById.get(edge.toNode);
    if (!from || !to) return "";

    const a = nodePoint(from, edge.fromSide);
    const b = nodePoint(to, edge.toSide);
    const dx = Math.min(Math.max(Math.abs(b.x - a.x) * 0.45, 80), 220);
    const dy = Math.min(Math.max(Math.abs(b.y - a.y) * 0.32, 70), 180);
    const c1 = { x: a.x, y: a.y };
    const c2 = { x: b.x, y: b.y };

    if (edge.fromSide === "right") c1.x += dx;
    if (edge.fromSide === "left") c1.x -= dx;
    if (edge.fromSide === "bottom") c1.y += dy;
    if (edge.fromSide === "top") c1.y -= dy;
    if (edge.toSide === "right") c2.x += dx;
    if (edge.toSide === "left") c2.x -= dx;
    if (edge.toSide === "bottom") c2.y += dy;
    if (edge.toSide === "top") c2.y -= dy;

    return `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`;
  }

  function boundsFor(nodes) {
    const pad = 100;
    const minX = Math.min(...nodes.map((node) => node.x)) - pad;
    const minY = Math.min(...nodes.map((node) => node.y)) - pad;
    const maxX = Math.max(...nodes.map((node) => node.x + node.width)) + pad;
    const maxY = Math.max(...nodes.map((node) => node.y + node.height)) + pad;
    return { minX, minY, maxX, maxY, width: maxX - minX, height: maxY - minY };
  }

  function initViewer(root) {
    const src = root.dataset.canvasSrc;
    if (!src) return;

    root.innerHTML = [
      '<div class="jc-toolbar">',
      '<span class="jc-title">故事结构画布</span>',
      '<button type="button" data-action="zoom-out">-</button>',
      '<button type="button" data-action="zoom-in">+</button>',
      '<button type="button" data-action="fit">重置</button>',
      '<span class="jc-hint">滚轮缩放，拖动画布</span>',
      '</div>',
      '<div class="jc-viewport" tabindex="0" aria-label="故事结构画布交互视图">',
      '<div class="jc-stage">',
      '<svg class="jc-edges" aria-hidden="true"></svg>',
      '<div class="jc-nodes"></div>',
      '</div>',
      '</div>',
    ].join("");

    const viewport = root.querySelector(".jc-viewport");
    const stage = root.querySelector(".jc-stage");
    const edgesSvg = root.querySelector(".jc-edges");
    const nodesLayer = root.querySelector(".jc-nodes");
    const toolbar = root.querySelector(".jc-toolbar");

    let state = { scale: 0.45, x: 0, y: 0 };
    let bounds = null;
    let dragging = null;

    function applyTransform() {
      stage.style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.scale})`;
    }

    function fit() {
      if (!bounds) return;
      const vw = viewport.clientWidth || 900;
      const vh = viewport.clientHeight || 560;
      const scale = Math.min(vw / bounds.width, vh / bounds.height, 0.72);
      state.scale = Math.max(scale, 0.18);
      state.x = (vw - bounds.width * state.scale) / 2 - bounds.minX * state.scale;
      state.y = (vh - bounds.height * state.scale) / 2 - bounds.minY * state.scale;
      applyTransform();
    }

    function zoomAt(delta, originX, originY) {
      const oldScale = state.scale;
      const nextScale = Math.min(Math.max(oldScale * delta, 0.18), 1.6);
      const wx = (originX - state.x) / oldScale;
      const wy = (originY - state.y) / oldScale;
      state.scale = nextScale;
      state.x = originX - wx * nextScale;
      state.y = originY - wy * nextScale;
      applyTransform();
    }

    function render(canvas) {
      const nodes = canvas.nodes || [];
      const edges = canvas.edges || [];
      const nodesById = new Map(nodes.map((node) => [node.id, node]));
      bounds = boundsFor(nodes);

      stage.style.width = `${bounds.width}px`;
      stage.style.height = `${bounds.height}px`;
      edgesSvg.setAttribute("viewBox", `${bounds.minX} ${bounds.minY} ${bounds.width} ${bounds.height}`);
      edgesSvg.style.left = `${bounds.minX}px`;
      edgesSvg.style.top = `${bounds.minY}px`;
      edgesSvg.style.width = `${bounds.width}px`;
      edgesSvg.style.height = `${bounds.height}px`;
      edgesSvg.innerHTML = [
        '<defs><marker id="jc-arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth"><path d="M2,2 L10,6 L2,10 Z" fill="#64748b"></path></marker></defs>',
        edges.map((edge) => {
          const color = colorFor(edge);
          const path = edgePath(edge, nodesById);
          if (!path) return "";
          const from = nodesById.get(edge.fromNode);
          const to = nodesById.get(edge.toNode);
          const a = nodePoint(from, edge.fromSide);
          const b = nodePoint(to, edge.toSide);
          const label = edge.label
            ? `<text x="${(a.x + b.x) / 2}" y="${(a.y + b.y) / 2 - 14}" text-anchor="middle">${escapeHtml(edge.label)}</text>`
            : "";
          return `<g style="${cssVars(color, "edge")}"><path d="${path}"></path>${label}</g>`;
        }).join(""),
      ].join("");

      nodesLayer.innerHTML = nodes.map((node) => {
        const color = colorFor(node);
        if (node.type === "group") {
          return `<section class="jc-group" style="left:${node.x}px;top:${node.y}px;width:${node.width}px;height:${node.height}px;${cssVars(color)}"><span>${escapeHtml(node.label || "")}</span></section>`;
        }
        const content = node.type === "text"
          ? markdownToHtml(node.text || "")
          : `<p>${escapeHtml(node.file || node.url || "")}</p>`;
        return `<article class="jc-node" style="left:${node.x}px;top:${node.y}px;width:${node.width}px;height:${node.height}px;${cssVars(color)}">${content}</article>`;
      }).join("");

      fit();
    }

    toolbar.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const rect = viewport.getBoundingClientRect();
      if (button.dataset.action === "zoom-in") zoomAt(1.18, rect.width / 2, rect.height / 2);
      if (button.dataset.action === "zoom-out") zoomAt(0.85, rect.width / 2, rect.height / 2);
      if (button.dataset.action === "fit") fit();
    });

    viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = viewport.getBoundingClientRect();
      zoomAt(event.deltaY < 0 ? 1.08 : 0.92, event.clientX - rect.left, event.clientY - rect.top);
    }, { passive: false });

    viewport.addEventListener("pointerdown", (event) => {
      if (event.target.closest("a")) return;
      dragging = { id: event.pointerId, x: event.clientX, y: event.clientY, ox: state.x, oy: state.y };
      viewport.setPointerCapture(event.pointerId);
      viewport.classList.add("is-dragging");
    });

    viewport.addEventListener("pointermove", (event) => {
      if (!dragging || dragging.id !== event.pointerId) return;
      state.x = dragging.ox + event.clientX - dragging.x;
      state.y = dragging.oy + event.clientY - dragging.y;
      applyTransform();
    });

    viewport.addEventListener("pointerup", (event) => {
      if (!dragging || dragging.id !== event.pointerId) return;
      dragging = null;
      viewport.classList.remove("is-dragging");
    });

    window.addEventListener("resize", fit);

    fetch(src)
      .then((response) => {
        if (!response.ok) throw new Error(`Failed to load ${src}`);
        return response.json();
      })
      .then(render)
      .catch((error) => {
        root.innerHTML = `<p class="jc-error">Canvas 加载失败：${escapeHtml(error.message)}</p>`;
      });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".json-canvas-viewer").forEach(initViewer);
  });
})();
