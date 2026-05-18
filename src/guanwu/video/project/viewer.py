from __future__ import annotations

import base64
import io
import json
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image, ImageDraw

from guanwu.video.project.artifacts import STAGE_ORDER

PHASES: list[dict[str, Any]] = [
    {
        "id": "parse",
        "title": "Parse",
        "stages": ["video.inspect", "frame.sample", "object.detect", "object.index", "object.attr"],
    },
    {
        "id": "lift",
        "title": "Lift",
        "stages": ["geometry.lift", "mesh.reconstruct"],
    },
    {
        "id": "infer",
        "title": "Infer",
        "stages": ["physics.dynamics", "relation.infer", "event.infer"],
    },
    {
        "id": "build",
        "title": "Build",
        "stages": ["scene.compose", "world.compose", "world.align"],
    },
    {
        "id": "export",
        "title": "Export",
        "stages": ["scene.export", "report.render"],
    },
    {
        "id": "publish",
        "title": "Publish",
        "stages": ["materialize", "catalog"],
    },
]

STAGE_TITLES: dict[str, str] = {
    "video.inspect": "video.inspect",
    "frame.sample": "frame.sample",
    "object.detect": "object.detect",
    "object.index": "object.index",
    "object.attr": "object.attr",
    "geometry.lift": "geometry.lift",
    "mesh.reconstruct": "mesh.reconstruct",
    "scene.compose": "scene.compose",
    "physics.dynamics": "physics.dynamics",
    "relation.infer": "relation.infer",
    "event.infer": "event.infer",
    "world.compose": "world.compose",
    "world.align": "world.align",
    "scene.export": "scene.export",
    "report.render": "report.render",
    "materialize": "materialize",
    "catalog": "catalog",
}

ROOT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Guanwu Project Viewer</title>
  <style>
    :root {
      --bg: #f2ede4;
      --paper: #fffaf2;
      --paper-2: #f7f1e7;
      --ink: #17313d;
      --muted: #64717c;
      --border: #d8d0c3;
      --accent: #c85a2b;
      --accent-2: #0f766e;
      --ok: #18794e;
      --warn: #b65d0e;
      --pending: #6b7280;
      --shadow: 0 12px 30px rgba(23, 49, 61, 0.08);
      --radius: 18px;
    }

    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; }
    body {
      font-family: "SF Pro Text", "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(15, 118, 110, 0.10), transparent 30%),
        radial-gradient(circle at top left, rgba(200, 90, 43, 0.12), transparent 28%),
        linear-gradient(180deg, #f6f1e8 0%, #efe8dc 100%);
      min-height: 100vh;
    }

    a { color: var(--accent-2); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, pre {
      font-family: "SFMono-Regular", "Menlo", "Monaco", monospace;
    }

    .app {
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
      padding: 24px;
    }

    .sidebar {
      position: sticky;
      top: 24px;
      align-self: start;
      background: rgba(255, 250, 242, 0.88);
      border: 1px solid rgba(216, 208, 195, 0.9);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 20px;
      backdrop-filter: blur(18px);
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 20px;
    }

    .brand h1 {
      margin: 0;
      font-size: 1.2rem;
      letter-spacing: 0.02em;
    }

    .muted {
      color: var(--muted);
    }

    .sidebar-block {
      margin-top: 18px;
      padding-top: 18px;
      border-top: 1px solid var(--border);
    }

    .phase-link {
      display: flex;
      align-items: center;
      justify-content: space-between;
      width: 100%;
      border: 1px solid var(--border);
      background: var(--paper);
      color: var(--ink);
      border-radius: 14px;
      padding: 10px 12px;
      margin: 8px 0;
      cursor: pointer;
      font-size: 0.95rem;
      transition: transform 120ms ease, border-color 120ms ease, background 120ms ease;
    }

    .phase-link:hover,
    .phase-link.active {
      transform: translateY(-1px);
      border-color: rgba(15, 118, 110, 0.45);
      background: #fffdf9;
    }

    .status-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    .status-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-radius: 12px;
      background: var(--paper-2);
      border: 1px solid rgba(216, 208, 195, 0.75);
      font-size: 0.9rem;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      border: 1px solid transparent;
      white-space: nowrap;
    }

    .badge.completed {
      color: var(--ok);
      background: rgba(24, 121, 78, 0.10);
      border-color: rgba(24, 121, 78, 0.18);
    }

    .badge.pending {
      color: var(--pending);
      background: rgba(107, 114, 128, 0.10);
      border-color: rgba(107, 114, 128, 0.15);
    }

    .badge.failed {
      color: #b42318;
      background: rgba(180, 35, 24, 0.10);
      border-color: rgba(180, 35, 24, 0.18);
    }

    .main {
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .hero {
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid rgba(216, 208, 195, 0.95);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 24px;
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
      align-items: start;
    }

    .hero h2 {
      margin: 0 0 8px;
      font-size: 2rem;
      line-height: 1.1;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }

    .hero-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 18px;
    }

    .button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      border-radius: 999px;
      padding: 10px 16px;
      border: 1px solid transparent;
      cursor: pointer;
      font-weight: 600;
      background: var(--accent);
      color: #fff;
    }

    .button.secondary {
      background: var(--paper);
      color: var(--ink);
      border-color: var(--border);
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
    }

    .metric-card {
      background: linear-gradient(180deg, #fffdf9 0%, #f7efe3 100%);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 14px 16px;
      min-height: 96px;
    }

    .metric-card .label {
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .metric-card .value {
      margin-top: 8px;
      font-size: 1.55rem;
      font-weight: 700;
      line-height: 1.1;
    }

    .section-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-top: 4px;
    }

    .section-title h3 {
      margin: 0;
      font-size: 1.35rem;
    }

    .stage-grid {
      display: grid;
      gap: 18px;
    }

    .stage-card {
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid rgba(216, 208, 195, 0.95);
      border-radius: 24px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .stage-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 18px 20px;
      border-bottom: 1px solid rgba(216, 208, 195, 0.8);
      background: linear-gradient(180deg, rgba(255,255,255,0.55) 0%, rgba(247,241,231,0.55) 100%);
    }

    .stage-head h4 {
      margin: 0;
      font-size: 1.05rem;
    }

    .stage-body {
      padding: 20px;
      display: grid;
      gap: 16px;
    }

    .empty {
      padding: 16px 18px;
      border-radius: 16px;
      background: #f7f1e7;
      border: 1px dashed var(--border);
      color: var(--muted);
    }

    .kv-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }

    .kv-card {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdf9;
      padding: 14px 16px;
    }

    .kv-card .k {
      font-size: 0.82rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }

    .kv-card .v {
      margin-top: 8px;
      font-size: 0.95rem;
      word-break: break-word;
    }

    .frame-viewer {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 18px;
      align-items: start;
    }

    .frame-controls {
      display: grid;
      gap: 10px;
    }

    .viewer-image {
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #e8ddd0;
    }

    .small-note {
      color: var(--muted);
      font-size: 0.9rem;
    }

    .table-wrap {
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdf9;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 480px;
    }

    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid rgba(216, 208, 195, 0.75);
      vertical-align: top;
      font-size: 0.93rem;
    }

    th {
      position: sticky;
      top: 0;
      background: #f8f2e8;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      font-size: 0.76rem;
    }

    details {
      border: 1px solid var(--border);
      border-radius: 16px;
      background: #fffdf9;
      padding: 12px 14px;
    }

    summary {
      cursor: pointer;
      font-weight: 600;
    }

    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 12px;
      background: #152832;
      color: #f3efe5;
      overflow: auto;
      font-size: 0.85rem;
    }

    iframe {
      width: 100%;
      height: 680px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: white;
    }

    .output-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    .chip-link,
    .chip-dir {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      border: 1px solid var(--border);
      background: #fffdf9;
      font-size: 0.88rem;
    }

    .chip-dir code,
    .chip-link code {
      background: transparent;
      padding: 0;
    }

    @media (max-width: 1100px) {
      .app {
        grid-template-columns: 1fr;
      }
      .sidebar {
        position: static;
      }
      .hero,
      .frame-viewer {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="brand">
        <div class="muted">Guanwu video pipeline</div>
        <h1 id="project-title">Project Viewer</h1>
        <div class="muted" id="project-root"></div>
      </div>

      <div class="sidebar-block">
        <div class="muted">Phases</div>
        <div id="phase-links"></div>
      </div>

      <div class="sidebar-block">
        <div class="muted">Stage status</div>
        <div class="status-list" id="status-list"></div>
      </div>
    </aside>

    <main class="main">
      <section class="hero">
        <div>
          <div class="muted">Offline project results viewer</div>
          <h2 id="hero-title">Loading viewer...</h2>
          <p id="hero-copy">The viewer mirrors the project-based outputs from the SPWM pipeline and lets you inspect each stage from one place.</p>
          <div class="hero-actions">
            <button class="button" id="refresh-button" type="button">Refresh</button>
            <a class="button secondary" id="project-inspect-link" href="/api/state" target="_blank" rel="noreferrer">Open state JSON</a>
          </div>
        </div>
        <div class="metric-grid" id="hero-metrics"></div>
      </section>

      <section>
        <div class="section-title">
          <h3 id="phase-title">Phase</h3>
          <span class="small-note" id="phase-subtitle"></span>
        </div>
      </section>

      <section class="stage-grid" id="phase-content"></section>
    </main>
  </div>

  <script>
    const stageTitles = {
      "video.inspect": "video.inspect",
      "frame.sample": "frame.sample",
      "object.detect": "object.detect",
      "object.index": "object.index",
      "object.attr": "object.attr",
      "geometry.lift": "geometry.lift",
      "mesh.reconstruct": "mesh.reconstruct",
      "scene.compose": "scene.compose",
      "physics.dynamics": "physics.dynamics",
      "relation.infer": "relation.infer",
      "event.infer": "event.infer",
      "world.compose": "world.compose",
      "world.align": "world.align",
      "scene.export": "scene.export",
      "report.render": "report.render",
      "materialize": "materialize",
      "catalog": "catalog",
    };

    let viewerState = null;

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function slugify(value) {
      return String(value || "").replaceAll(".", "-");
    }

    function titleCase(value) {
      return String(value || "").replace(/(^|\\s|[.-])([a-z])/g, (_, a, b) => `${a}${b.toUpperCase()}`);
    }

    function stageStatus(stage) {
      return viewerState?.statuses?.[stage]?.status || "pending";
    }

    function statusBadge(status) {
      const normalized = String(status || "pending").toLowerCase();
      return `<span class="badge ${normalized}">${escapeHtml(normalized)}</span>`;
    }

    function outputMeta(stage) {
      return viewerState?.artifacts?.[stage]?.outputs_meta || {};
    }

    async function fetchJson(url) {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    }

    function asJsonUrl(path) {
      return `/api/json?path=${encodeURIComponent(path || "")}`;
    }

    function asFileUrl(path) {
      return `/api/file?path=${encodeURIComponent(path || "")}`;
    }

    function asDetectRenderUrl(detectionsPath, fallbackPath) {
      const params = new URLSearchParams();
      if (detectionsPath) {
        params.set("path", detectionsPath);
      }
      if (fallbackPath) {
        params.set("fallback", fallbackPath);
      }
      params.set("t", String(Date.now()));
      return `/api/object-detect/render?${params.toString()}`;
    }

    function metricCards(metrics) {
      if (!metrics.length) {
        return `<div class="metric-card"><div class="label">No metrics</div><div class="value">-</div></div>`;
      }
      return metrics.map((item) => `
        <div class="metric-card">
          <div class="label">${escapeHtml(item.label)}</div>
          <div class="value">${escapeHtml(item.value)}</div>
        </div>
      `).join("");
    }

    function kvGrid(items) {
      if (!items.length) {
        return "";
      }
      return `
        <div class="kv-grid">
          ${items.map((item) => `
            <div class="kv-card">
              <div class="k">${escapeHtml(item.key)}</div>
              <div class="v">${escapeHtml(item.value)}</div>
            </div>
          `).join("")}
        </div>
      `;
    }

    function renderTable(rows) {
      if (!rows || !rows.length) {
        return `<div class="empty">No rows available.</div>`;
      }
      const headers = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));
      return `
        <div class="table-wrap">
          <table>
            <thead>
              <tr>${headers.map((key) => `<th>${escapeHtml(key)}</th>`).join("")}</tr>
            </thead>
            <tbody>
              ${rows.map((row) => `
                <tr>
                  ${headers.map((key) => {
                    const value = row[key];
                    const rendered = value && typeof value === "object" ? JSON.stringify(value) : (value ?? "");
                    return `<td>${escapeHtml(rendered)}</td>`;
                  }).join("")}
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>
      `;
    }

    function renderPre(title, payload) {
      if (payload == null) {
        return "";
      }
      return `
        <details>
          <summary>${escapeHtml(title)}</summary>
          <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </details>
      `;
    }

    function renderOutputLinks(meta) {
      const entries = Object.entries(meta || {});
      if (!entries.length) {
        return "";
      }
      return `
        <div class="output-links">
          ${entries.map(([key, value]) => {
            if (!value || !value.exists) {
              return "";
            }
            if (value.is_dir) {
              return `<span class="chip-dir">${escapeHtml(key)} <code>${escapeHtml(value.path)}</code></span>`;
            }
            return `<a class="chip-link" href="${escapeHtml(value.file_url)}" target="_blank" rel="noreferrer">${escapeHtml(key)}</a>`;
          }).join("")}
        </div>
      `;
    }

    function stageCard(stage) {
      const artifact = viewerState?.artifacts?.[stage];
      const summary = artifact?.summary || {};
      const summaryEntries = Object.entries(summary).map(([key, value]) => ({
        key,
        value: Array.isArray(value) || (value && typeof value === "object") ? JSON.stringify(value) : String(value),
      }));
      return `
        <article class="stage-card">
          <div class="stage-head">
            <h4>${escapeHtml(stageTitles[stage] || stage)}</h4>
            ${statusBadge(stageStatus(stage))}
          </div>
          <div class="stage-body">
            ${renderOutputLinks(outputMeta(stage))}
            ${summaryEntries.length ? kvGrid(summaryEntries) : ""}
            <div id="content-${slugify(stage)}">
              <div class="empty">Loading stage data...</div>
            </div>
          </div>
        </article>
      `;
    }

    function pendingStageCard(stage) {
      return `
        <article class="stage-card">
          <div class="stage-head">
            <h4>${escapeHtml(stageTitles[stage] || stage)}</h4>
            ${statusBadge(stageStatus(stage))}
          </div>
          <div class="stage-body">
            <div class="empty">
              Stage <code>${escapeHtml(stage)}</code> has not completed yet.
              Run <code>guanwu video step ${escapeHtml(stage)} ${escapeHtml(viewerState.project_root)}</code> first.
            </div>
          </div>
        </article>
      `;
    }

    function activePhaseId() {
      const raw = window.location.hash.replace(/^#/, "");
      const phases = viewerState?.phases?.map((item) => item.id) || [];
      return phases.includes(raw) ? raw : (viewerState?.phases?.[0]?.id || "parse");
    }

    function updateHero(state) {
      document.getElementById("project-title").textContent = state.project_name || "Project Viewer";
      document.getElementById("project-root").textContent = state.project_root || "";
      document.getElementById("hero-title").textContent = state.project_name || "Project Viewer";
      document.getElementById("hero-copy").textContent =
        state.project?.input_video
          ? `Inspecting ${state.project.input_video}`
          : "Inspect the project outputs stage by stage from one place.";

      const completed = Object.values(state.statuses || {}).filter((item) => item.status === "completed").length;
      const total = Object.keys(state.statuses || {}).length;
      const metrics = [
        { label: "Project ID", value: state.project?.project_id || state.manifest?.project_id || "-" },
        { label: "Completed", value: `${completed}/${total}` },
        { label: "Latest stage", value: state.latest_completed_stage || "-" },
        { label: "Input video", value: state.project?.input_video ? titleCase(PathBasename(state.project.input_video)) : "-" },
      ];
      document.getElementById("hero-metrics").innerHTML = metricCards(metrics);
    }

    function PathBasename(path) {
      return String(path || "").split(/[\\\\/]/).filter(Boolean).pop() || "";
    }

    function updateSidebar(state) {
      const links = document.getElementById("phase-links");
      links.innerHTML = state.phases.map((phase) => {
        const done = phase.stages.filter((stage) => stageStatus(stage) === "completed").length;
        const active = phase.id === activePhaseId() ? "active" : "";
        return `
          <button class="phase-link ${active}" type="button" data-phase="${escapeHtml(phase.id)}">
            <span>${escapeHtml(phase.title)}</span>
            <span class="muted">${done}/${phase.stages.length}</span>
          </button>
        `;
      }).join("");
      links.querySelectorAll(".phase-link").forEach((button) => {
        button.addEventListener("click", () => {
          window.location.hash = button.dataset.phase;
        });
      });

      const statusList = document.getElementById("status-list");
      statusList.innerHTML = state.stage_order.map((stage) => `
        <div class="status-item">
          <span>${escapeHtml(stageTitles[stage] || stage)}</span>
          ${statusBadge(stageStatus(stage))}
        </div>
      `).join("");
    }

    async function renderPhase() {
      const phase = viewerState.phases.find((item) => item.id === activePhaseId()) || viewerState.phases[0];
      document.getElementById("phase-title").textContent = phase.title;
      document.getElementById("phase-subtitle").textContent = `${phase.stages.length} stage${phase.stages.length === 1 ? "" : "s"}`;

      const container = document.getElementById("phase-content");
      container.innerHTML = phase.stages.map((stage) => {
        return stageStatus(stage) === "completed" ? stageCard(stage) : pendingStageCard(stage);
      }).join("");

      updateSidebar(viewerState);

      for (const stage of phase.stages) {
        if (stageStatus(stage) !== "completed") {
          continue;
        }
        try {
          await hydrateStage(stage);
        } catch (error) {
          const node = document.getElementById(`content-${slugify(stage)}`);
          if (node) {
            node.innerHTML = `<div class="empty">Failed to render stage: ${escapeHtml(error.message || error)}</div>`;
          }
        }
      }
    }

    async function hydrateStage(stage) {
      switch (stage) {
        case "video.inspect":
          return await renderVideoInspect(stage);
        case "frame.sample":
          return await renderFrameSample(stage);
        case "object.detect":
          return await renderObjectDetect(stage);
        case "object.index":
          return await renderObjectIndex(stage);
        case "object.attr":
          return await renderObjectAttr(stage);
        case "geometry.lift":
          return await renderGeometryLift(stage);
        case "mesh.reconstruct":
          return await renderMeshReconstruct(stage);
        case "scene.compose":
          return await renderSceneCompose(stage);
        case "physics.dynamics":
          return await renderSimpleJsonTable(stage, "physics_dynamics");
        case "relation.infer":
          return await renderSimpleJsonTable(stage, "relations");
        case "event.infer":
          return await renderEventInfer(stage);
        case "world.compose":
          return await renderWorldCompose(stage);
        case "world.align":
          return await renderWorldAlign(stage);
        case "scene.export":
          return await renderSceneExport(stage);
        case "report.render":
          return await renderReport(stage);
        case "materialize":
          return await renderJsonArtifact(stage, "materialize_report");
        case "catalog":
          return await renderJsonArtifact(stage, "catalog_stats");
        default:
          return await renderFallback(stage);
      }
    }

    async function renderVideoInspect(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const meta = outputMeta(stage).video_metadata;
      const metadata = meta?.json_url ? await fetchJson(meta.json_url) : null;
      const firstFrame = outputMeta("frame.sample").first_frame;
      const metrics = [];
      if (metadata) {
        metrics.push({ label: "Frames", value: metadata.frame_count ?? "-" });
        metrics.push({ label: "FPS", value: metadata.fps ?? "-" });
        metrics.push({ label: "Width", value: metadata.width ?? "-" });
        metrics.push({ label: "Height", value: metadata.height ?? "-" });
      }
      container.innerHTML = `
        ${metrics.length ? metricCards(metrics) : `<div class="empty">No video metadata found.</div>`}
        ${firstFrame?.file_url ? `<img class="viewer-image" src="${escapeHtml(firstFrame.file_url)}" alt="First frame">` : ""}
        ${renderPre("video.inspect JSON", metadata)}
      `;
    }

    async function renderFrameSample(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const indexMeta = outputMeta(stage).frame_index;
      const frames = indexMeta?.json_url ? await fetchJson(indexMeta.json_url) : [];
      const preview = Array.isArray(frames) ? frames.slice(0, 40) : [];
      container.innerHTML = `
        ${metricCards([{ label: "Sampled frames", value: Array.isArray(frames) ? frames.length : 0 }])}
        ${renderTable(preview)}
        ${renderPre("frame.sample JSON", frames)}
      `;
    }

    async function renderObjectDetect(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const summaryMeta = outputMeta(stage).summary;
      const summary = summaryMeta?.json_url ? await fetchJson(summaryMeta.json_url) : null;
      const frames = summary?.frames || [];
      if (!frames.length) {
        container.innerHTML = `<div class="empty">No object.detect frames found.</div>`;
        return;
      }

      const sliderId = `slider-${slugify(stage)}`;
      const metaId = `meta-${slugify(stage)}`;
      const imageId = `image-${slugify(stage)}`;
      const tableId = `table-${slugify(stage)}`;

      container.innerHTML = `
        ${metricCards([
          { label: "Frames processed", value: frames.length },
          { label: "Latest instances", value: summary?.latest_instance_count ?? "-" },
        ])}
        <div class="frame-controls">
          <label for="${sliderId}">Frame browser</label>
          <input id="${sliderId}" type="range" min="0" max="${Math.max(frames.length - 1, 0)}" value="0" step="1">
          <div class="small-note" id="${metaId}"></div>
        </div>
        <div class="frame-viewer">
          <div>
            <img class="viewer-image" id="${imageId}" alt="Detection overlay">
          </div>
          <div id="${tableId}"></div>
        </div>
      `;

      const slider = document.getElementById(sliderId);
      const meta = document.getElementById(metaId);
      const image = document.getElementById(imageId);
      const table = document.getElementById(tableId);

      const updateFrame = async () => {
        const idx = Number(slider.value || 0);
        const entry = frames[idx];
        meta.textContent = `Frame ${entry.frame_idx} at ${Number(entry.timestamp || 0).toFixed(3)}s`;
        image.src = asDetectRenderUrl(entry.detections, entry.overlay);
        const detections = entry.detections ? await fetchJson(asJsonUrl(entry.detections)) : null;
        const instances = detections?.instances || [];
        const rows = instances.map((inst) => ({
          object_id: inst.object_id || "",
          label: inst.concept_label || "",
          score: typeof inst.score === "number" ? inst.score.toFixed(3) : inst.score ?? "",
          bbox: Array.isArray(inst.bbox) ? inst.bbox.map((v) => Number(v).toFixed(1)).join(", ") : "",
          kind: inst.segment_kind || "",
        }));
        table.innerHTML = `
          ${metricCards([{ label: "Instances", value: instances.length }])}
          ${renderTable(rows)}
        `;
      };

      slider.addEventListener("input", updateFrame);
      await updateFrame();
    }

    async function renderObjectIndex(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const objectsMeta = outputMeta(stage).objects;
      const objects = objectsMeta?.json_url ? await fetchJson(objectsMeta.json_url) : [];
      const rows = (Array.isArray(objects) ? objects : []).map((item) => {
        const frames = Array.isArray(item.frames) ? item.frames : [];
        const frameIds = frames.map((frame) => frame.frame_idx).filter((value) => Number.isFinite(value));
        return {
          object_id: item.object_id || "",
          label: item.label || "",
          segment_kind: item.segment_kind || "",
          frames: frames.length,
          first_frame: frameIds.length ? Math.min(...frameIds) : "",
          last_frame: frameIds.length ? Math.max(...frameIds) : "",
        };
      });
      container.innerHTML = `
        ${metricCards([{ label: "Objects", value: rows.length }])}
        ${renderTable(rows)}
      `;
    }

    async function renderObjectAttr(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const attrsMeta = outputMeta(stage).object_attrs;
      const attrs = attrsMeta?.json_url ? await fetchJson(attrsMeta.json_url) : {};
      const rows = Object.entries(attrs || {}).map(([objectId, value]) => {
        const materials = Array.isArray(value.material_candidates) ? value.material_candidates : [];
        const mass = Array.isArray(value.mass_range_kg) ? value.mass_range_kg.join(" - ") : "";
        return {
          object_id: objectId,
          class_name: value.class_name || "",
          movable: value.is_movable ?? "",
          rigid_body: value.is_rigid_body ?? "",
          top_material: materials[0]?.name || "",
          mass_range_kg: mass,
          confidence: value.confidence ?? "",
        };
      });
      container.innerHTML = `
        ${metricCards([{ label: "Attributed objects", value: rows.length }])}
        ${renderTable(rows)}
      `;
    }

    async function renderGeometryLift(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const summaryMeta = outputMeta(stage).summary;
      const summary = summaryMeta?.json_url ? await fetchJson(summaryMeta.json_url) : null;
      const frames = summary?.frames || [];
      const latestObjects = summary?.latest_objects || [];
      const cameraPath = outputMeta(stage).camera_trajectory;
      const cameras = cameraPath?.json_url ? await fetchJson(cameraPath.json_url) : [];
      const rows = latestObjects.map((item) => {
        const pos = item?.geometry?.pose_3d?.position || item?.geometry?.centroid_3d || [];
        return {
          object_id: item.object_id || "",
          label: item.label || "",
          x: Array.isArray(pos) ? pos[0] ?? "" : "",
          y: Array.isArray(pos) ? pos[1] ?? "" : "",
          z: Array.isArray(pos) ? pos[2] ?? "" : "",
        };
      });
      container.innerHTML = `
        ${metricCards([
          { label: "Frames", value: frames.length },
          { label: "Latest objects", value: latestObjects.length },
          { label: "Camera samples", value: Array.isArray(cameras) ? cameras.length : 0 },
        ])}
        ${renderTable(rows)}
        ${renderPre("geometry.lift summary", summary)}
      `;
    }

    async function renderMeshReconstruct(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const meshesMeta = outputMeta(stage).sam3d_meshes;
      const meshes = meshesMeta?.json_url ? await fetchJson(meshesMeta.json_url) : {};
      const rows = Object.entries(meshes || {}).map(([objectId, value]) => ({
        object_id: objectId,
        source: value.source || "",
        quality: value.quality || "",
        frame_idx: value.reconstruction_frame_idx ?? "",
        mesh_path: value.mesh_path || (Array.isArray(value.files) ? (value.files[0]?.path || "") : ""),
      }));
      container.innerHTML = `
        ${metricCards([{ label: "Meshes", value: rows.length }])}
        ${renderTable(rows)}
      `;
    }

    async function renderSceneCompose(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const manifestMeta = outputMeta(stage).scene_manifest;
      const manifest = manifestMeta?.json_url ? await fetchJson(manifestMeta.json_url) : null;
      container.innerHTML = `
        ${renderPre("scene.compose manifest", manifest)}
      `;
    }

    async function renderSimpleJsonTable(stage, outputKey) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const meta = outputMeta(stage)[outputKey];
      const payload = meta?.json_url ? await fetchJson(meta.json_url) : null;
      const rows = Array.isArray(payload)
        ? payload.map((item) => flattenRow(item))
        : Object.entries(payload || {}).map(([key, value]) => ({ key, ...(typeof value === "object" && value ? value : { value }) }));
      container.innerHTML = `
        ${renderTable(rows)}
        ${renderPre(`${stage} JSON`, payload)}
      `;
    }

    function flattenRow(item) {
      const row = {};
      for (const [key, value] of Object.entries(item || {})) {
        row[key] = value && typeof value === "object" ? JSON.stringify(value) : value;
      }
      return row;
    }

    async function renderEventInfer(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const eventsMeta = outputMeta(stage).events;
      const events = eventsMeta?.json_url ? await fetchJson(eventsMeta.json_url) : [];
      const rows = (Array.isArray(events) ? events : []).map((item) => flattenRow(item));
      container.innerHTML = `
        ${metricCards([{ label: "Events", value: rows.length }])}
        ${renderTable(rows)}
      `;
    }

    async function renderWorldCompose(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const worldMeta = outputMeta(stage).world_state_raw;
      const world = worldMeta?.json_url ? await fetchJson(worldMeta.json_url) : null;
      const rows = (world?.active_objects || []).map((item) => ({
        object_id: item.object_id || "",
        label: item.label || "",
        visibility: item.state?.visibility || "",
      }));
      container.innerHTML = `
        ${metricCards([
          { label: "Objects", value: rows.length },
          { label: "Relations", value: Array.isArray(world?.relations) ? world.relations.length : 0 },
          { label: "Events", value: Array.isArray(world?.events) ? world.events.length : 0 },
        ])}
        ${renderTable(rows)}
        ${renderPre("world.compose JSON", world)}
      `;
    }

    async function renderWorldAlign(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const worldMeta = outputMeta(stage).world_state_aligned;
      const world = worldMeta?.json_url ? await fetchJson(worldMeta.json_url) : null;
      const rows = (world?.objects || []).map((item) => {
        const pos = item?.geometry?.pose_3d?.position || [];
        return {
          object_id: item.object_id || "",
          label: item.label || "",
          visibility: item.state?.visibility || "",
          x: Array.isArray(pos) ? pos[0] ?? "" : "",
          y: Array.isArray(pos) ? pos[1] ?? "" : "",
          z: Array.isArray(pos) ? pos[2] ?? "" : "",
        };
      });
      container.innerHTML = `
        ${metricCards([
          { label: "Objects", value: rows.length },
          { label: "Relations", value: Array.isArray(world?.relations) ? world.relations.length : 0 },
          { label: "Recent events", value: Array.isArray(world?.events_recent) ? world.events_recent.length : 0 },
        ])}
        ${renderTable(rows)}
        ${renderPre("world.align JSON", world)}
      `;
    }

    async function renderSceneExport(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const exportMeta = outputMeta(stage).export;
      const conversionMeta = outputMeta(stage).conversion_report;
      const payload = exportMeta?.json_url
        ? await fetchJson(exportMeta.json_url)
        : (conversionMeta?.json_url ? await fetchJson(conversionMeta.json_url) : null);
      container.innerHTML = `
        ${payload ? renderPre("scene.export JSON", payload) : `<div class="empty">No JSON export payload found for this stage. Use the file links above for binary outputs such as USDC.</div>`}
      `;
    }

    async function renderReport(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const htmlMeta = outputMeta(stage).index_html;
      if (!htmlMeta?.file_url) {
        container.innerHTML = `<div class="empty">No report HTML found.</div>`;
        return;
      }
      container.innerHTML = `
        <iframe src="${escapeHtml(htmlMeta.file_url)}" title="report.render"></iframe>
      `;
    }

    async function renderJsonArtifact(stage, outputKey) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const meta = outputMeta(stage)[outputKey];
      const payload = meta?.json_url ? await fetchJson(meta.json_url) : null;
      container.innerHTML = payload ? renderPre(`${stage} JSON`, payload) : `<div class="empty">No JSON payload found.</div>`;
    }

    async function renderFallback(stage) {
      const container = document.getElementById(`content-${slugify(stage)}`);
      const artifact = viewerState.artifacts?.[stage];
      container.innerHTML = renderPre(`${stage} artifact`, artifact);
    }

    async function boot() {
      viewerState = await fetchJson("/api/state");
      updateHero(viewerState);
      updateSidebar(viewerState);
      await renderPhase();
    }

    document.getElementById("refresh-button").addEventListener("click", () => {
      window.location.reload();
    });

    window.addEventListener("hashchange", renderPhase);
    boot().catch((error) => {
      const container = document.getElementById("phase-content");
      container.innerHTML = `<div class="empty">Failed to load viewer: ${escapeHtml(error.message || error)}</div>`;
    });
  </script>
</body>
</html>
"""


def _read_json(path: Path) -> Any | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_project_toml(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.is_file():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_within_project(project_root: Path, raw_path: str | Path) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Path escapes project root: {candidate}") from exc
    return candidate


def _output_meta(project_root: Path, raw_path: str | None) -> dict[str, Any] | None:
    if not raw_path:
        return None
    try:
        resolved = _resolve_within_project(project_root, raw_path)
    except HTTPException:
        return {
            "path": str(raw_path),
            "exists": False,
            "is_dir": False,
            "file_url": None,
            "json_url": None,
        }
    exists = resolved.exists()
    is_dir = exists and resolved.is_dir()
    encoded = quote(str(resolved), safe="")
    return {
        "path": str(resolved),
        "exists": exists,
        "is_dir": is_dir,
        "file_url": f"/api/file?path={encoded}" if exists and not is_dir else None,
        "json_url": f"/api/json?path={encoded}" if exists and resolved.suffix.lower() == ".json" else None,
    }


def _default_statuses() -> dict[str, dict[str, Any]]:
    return {
        stage: {
            "stage": stage,
            "status": "pending",
            "last_run_at": None,
            "error": None,
            "inputs_hash": None,
            "params_hash": None,
        }
        for stage in STAGE_ORDER
    }


def _load_state_payload(project_root: Path) -> dict[str, Any]:
    state_dir = project_root / "state"
    manifest = _read_json(state_dir / "manifest.json") or {}
    project_payload = _read_project_toml(project_root / "project.toml")
    statuses = _default_statuses()
    statuses.update(_read_json(state_dir / "stage_status.json") or {})
    raw_artifacts = _read_json(state_dir / "artifacts.json") or {}

    artifacts: dict[str, Any] = {}
    for stage, record in raw_artifacts.items():
        outputs = record.get("outputs") if isinstance(record, dict) else {}
        output_meta = {}
        if isinstance(outputs, dict):
            for key, value in outputs.items():
                output_meta[key] = _output_meta(project_root, str(value))
        payload = dict(record) if isinstance(record, dict) else {"stage": stage}
        payload["outputs_meta"] = output_meta
        artifacts[stage] = payload

    latest_completed_stage = None
    for stage in STAGE_ORDER:
        if statuses.get(stage, {}).get("status") == "completed":
            latest_completed_stage = stage

    return {
        "project_root": str(project_root),
        "project_name": project_payload.get("project", {}).get("name") or manifest.get("project_id") or project_root.name,
        "project": project_payload.get("project", {}),
        "manifest": manifest,
        "statuses": statuses,
        "artifacts": artifacts,
        "phases": PHASES,
        "stage_order": STAGE_ORDER,
        "latest_completed_stage": latest_completed_stage,
    }


def _image_from_detections(detections: dict[str, Any], fallback_path: Path | None) -> Image.Image:
    image_b64 = detections.get("image_b64")
    if isinstance(image_b64, str) and image_b64:
        raw = image_b64.split(",", 1)[-1] if "," in image_b64 else image_b64
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    if fallback_path and fallback_path.exists() and fallback_path.is_file():
        return Image.open(fallback_path).convert("RGB")
    return Image.new("RGB", (640, 360), color=(238, 232, 223))


def _render_object_detect_overlay(detections: dict[str, Any], fallback_path: Path | None) -> bytes:
    overlay = np.array(_image_from_detections(detections, fallback_path), dtype=np.float32)
    instances = detections.get("instances", [])
    palette = [
        (255, 90, 80),
        (80, 176, 255),
        (64, 210, 140),
        (255, 188, 82),
        (196, 104, 255),
        (80, 212, 208),
        (255, 136, 196),
        (160, 214, 86),
        (255, 156, 92),
        (94, 126, 255),
    ]

    try:
        from pycocotools import mask as mask_util
    except ImportError:
        mask_util = None

    for index, instance in enumerate(instances):
        color = palette[index % len(palette)]
        mask_rle = instance.get("mask_rle")
        mask_drawn = False

        if mask_util is not None and mask_rle:
            try:
                rle = json.loads(mask_rle) if isinstance(mask_rle, str) else mask_rle
                binary_mask = mask_util.decode(rle).astype(bool)
                if binary_mask.any():
                    for channel, channel_value in enumerate(color):
                        overlay[:, :, channel][binary_mask] = (
                            overlay[:, :, channel][binary_mask] * 0.4 + channel_value * 0.6
                        )
                    mask_drawn = True
            except Exception:
                mask_drawn = False

        if not mask_drawn:
            bbox = instance.get("bbox", [])
            if isinstance(bbox, list) and len(bbox) >= 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                x1 = max(0, min(overlay.shape[1] - 1, x1))
                x2 = max(0, min(overlay.shape[1], x2))
                y1 = max(0, min(overlay.shape[0] - 1, y1))
                y2 = max(0, min(overlay.shape[0], y2))
                if x2 > x1 and y2 > y1:
                    overlay[y1:y2, x1 : x1 + 2, :] = color
                    overlay[y1:y2, max(x1, x2 - 2) : x2, :] = color
                    overlay[y1 : y1 + 2, x1:x2, :] = color
                    overlay[max(y1, y2 - 2) : y2, x1:x2, :] = color

        rendered = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(rendered)
        bbox = instance.get("bbox", [0, 0, 0, 0])
        x1 = int(float(bbox[0])) if len(bbox) >= 1 else 0
        y1 = int(float(bbox[1])) if len(bbox) >= 2 else 0
        label = instance.get("concept_label") or instance.get("object_id") or "object"
        score = instance.get("score")
        if isinstance(score, (int, float)):
            label = f"{label} {score:.2f}"
        tx = max(0, x1)
        ty = max(0, y1 - 18)
        draw.rectangle([tx, ty, tx + max(60, len(label) * 7), ty + 16], fill=color)
        draw.text((tx + 4, ty + 2), label, fill=(16, 24, 32))
        overlay = np.array(rendered, dtype=np.float32)

    final_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    buffer = io.BytesIO()
    final_image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_project_viewer_app(project_root: str | Path) -> FastAPI:
    root = Path(project_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Project directory not found: {root}")

    app = FastAPI(title="Guanwu Project Viewer", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return ROOT_HTML

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(_load_state_payload(root))

    @app.get("/api/json")
    def read_json_file(path: str = Query(..., description="Absolute or project-relative JSON path")) -> JSONResponse:
        resolved = _resolve_within_project(root, path)
        payload = _read_json(resolved)
        if payload is None:
            raise HTTPException(status_code=404, detail=f"JSON file not found or unreadable: {resolved}")
        return JSONResponse(payload)

    @app.get("/api/file")
    def read_file(path: str = Query(..., description="Absolute or project-relative file path")) -> FileResponse:
        resolved = _resolve_within_project(root, path)
        if not resolved.exists() or not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {resolved}")
        return FileResponse(resolved)

    @app.get("/api/object-detect/render")
    def render_object_detect(
        path: str = Query(..., description="Absolute or project-relative detections.json path"),
        fallback: str | None = Query(None, description="Optional fallback image path"),
    ) -> Response:
        detections_path = _resolve_within_project(root, path)
        detections = _read_json(detections_path)
        if not isinstance(detections, dict):
            raise HTTPException(status_code=404, detail=f"Detections JSON not found: {detections_path}")
        fallback_path = _resolve_within_project(root, fallback) if fallback else None
        png_bytes = _render_object_detect_overlay(detections, fallback_path)
        return Response(content=png_bytes, media_type="image/png")

    return app


def serve_project_viewer(
    project_root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8811,
) -> None:
    app = create_project_viewer_app(project_root)
    uvicorn.run(app, host=host, port=port, log_level="info")
