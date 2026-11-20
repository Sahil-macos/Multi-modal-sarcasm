(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (p) => (p == null ? "–" : `${(p * 100).toFixed(1)}%`);

  const VIDEO = /\.(mp4|mov|avi|mkv|webm|m4v|mpe?g|wmv)$/i;
  const AUDIO = /\.(wav|mp3|m4a|flac|ogg|aac|opus|wma)$/i;
  const IMAGE = /\.(png|jpe?g|webp|bmp|tiff?)$/i;
  const kindOf = (n) => (VIDEO.test(n) ? "video" : AUDIO.test(n) ? "audio" : "document");

  let file = null, previewUrl = null;

  // ---------- health ----------
  async function health() {
    try {
      const h = await (await fetch("/api/health")).json();
      let pending = false;
      document.querySelectorAll("#health .pill").forEach((p) => {
        const s = h.models[p.dataset.k] || "";
        p.className = "pill " + (s === "loaded" ? "ok" : s.startsWith("error") ? "err"
          : s.startsWith("not trained") ? "" : "load");
        p.title = s;
        if (s === "not loaded") pending = true;
      });
      if (pending) setTimeout(health, 3000);
    } catch { setTimeout(health, 5000); }
  }
  health();

  // ---------- file input ----------
  const drop = $("drop"), input = $("file");
  function setFile(f) {
    file = f;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    $("preview").innerHTML = "";
    $("file-chip").hidden = !f;
    drop.hidden = !!f;
    if (!f) { input.value = ""; return; }
    const k = kindOf(f.name);
    $("file-kind").textContent = k;
    $("file-name").textContent = `${f.name} · ${(f.size / 1048576).toFixed(2)} MB`;
    previewUrl = URL.createObjectURL(f);
    if (k === "video") $("preview").innerHTML = `<video src="${previewUrl}" controls muted></video>`;
    else if (k === "audio") $("preview").innerHTML = `<audio src="${previewUrl}" controls></audio>`;
    else if (IMAGE.test(f.name)) $("preview").innerHTML = `<img src="${previewUrl}" alt="">`;
  }
  input.addEventListener("change", () => setFile(input.files[0] || null));
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  $("file-clear").addEventListener("click", () => setFile(null));
  document.querySelectorAll("[data-ex]").forEach((b) => b.addEventListener("click", () => { $("text").value = b.dataset.ex; }));
  $("reset").addEventListener("click", () => {
    $("text").value = ""; setFile(null); $("err").hidden = true;
    $("res").hidden = true; $("empty").hidden = false;
  });

  // ---------- analyze ----------
  function stages(k) {
    const s = { document: ["Uploading…", "Extracting document text…", "Running RoBERTa…", "Fusing predictions…"],
      audio: ["Uploading…", "Decoding audio…", "Transcribing speech…", "Running RoBERTa + HuBERT…", "Fusing predictions…"],
      video: ["Uploading…", "Sampling frames…", "Extracting audio track…", "Transcribing speech…", "Reading on-screen text (OCR)…",
        "Running RoBERTa + HuBERT + VideoMAE…", "Fusing predictions…"],
      text: ["Running RoBERTa…", "Fusing predictions…"] };
    return s[k];
  }

  $("go").addEventListener("click", async () => {
    const text = $("text").value.trim();
    $("err").hidden = true;
    if (!text && !file) { showErr("Enter some text or choose a file."); return; }
    const fd = new FormData();
    if (text) fd.append("text", text);
    if (file) fd.append("file", file);
    fd.append("transcribe", $("opt-stt").checked);
    fd.append("ocr", $("opt-ocr").checked);

    const st = stages(file ? kindOf(file.name) : "text");
    let i = 0;
    $("stage").textContent = st[0];
    const timer = setInterval(() => { if (i < st.length - 1) $("stage").textContent = st[++i]; }, file ? 4000 : 900);
    $("go").disabled = true; $("loading").hidden = false; $("empty").hidden = true; $("res").hidden = true;
    try {
      const r = await fetch("/api/analyze", { method: "POST", body: fd });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(data.detail || `Request failed (${r.status})`);
      render(data);
    } catch (e) {
      showErr(e.message);
      $("empty").hidden = false;
    } finally {
      clearInterval(timer);
      $("go").disabled = false; $("loading").hidden = true;
    }
  });

  function showErr(m) { $("err").textContent = m; $("err").hidden = false; }

  // ---------- render ----------
  const CLS = { SARCASTIC: "sarc", NOT_SARCASTIC: "lit", UNCERTAIN: "unc", INSUFFICIENT_INPUT: "unc" };
  const MODEL_INFO = {
    RoBERTa: "Text classifier", HuBERT: "Speech prosody classifier", VideoMAE: "Visual (facial/gesture) classifier",
  };

  function render(d) {
    const r = d.result;
    const v = $("verdict");
    v.className = "card verdict " + (CLS[r.decision] || "");
    $("decision").textContent = r.decision.replace(/_/g, " ");
    $("score").textContent = r.sarcasm_score ?? "–";
    $("g-fg").style.strokeDashoffset = 157 * (1 - (r.sarcasm_score ?? 0) / 100);
    $("conf").textContent = r.confidence ?? "–";
    $("level").textContent = r.level ?? "–";
    $("agree").textContent = r.agreement ?? "–";
    $("expl").innerHTML = (r.explanation || []).map((e) => `<li>${esc(e)}</li>`).join("");

    // model cards (always show all three so missing modalities are visible)
    const preds = d.predictions || [];
    const cards = preds.map((p) => {
      const s = p.probability >= p.threshold;
      const foot = [
        `threshold ${p.threshold}`,
        p.chunks > 1 ? `${p.chunks} windows` : null,
        p.duration_s != null ? `${p.duration_s}s audio` : null,
        p.frames_used ? `${p.frames_used} frames` : null,
        `${p.latency_s}s`,
      ].filter(Boolean).join(" · ");
      return `<div class="model">
        <div class="m-head"><span class="m-name">${esc(p.model)}</span><span class="m-src">${esc(p.source)}</span></div>
        <div class="m-prob">${pct(p.probability)}</div>
        <div class="m-lab ${s ? "s" : "n"}">${s ? "Sarcastic" : "Not sarcastic"}</div>
        <div class="bar"><i class="${s ? "s" : "n"}" style="width:${p.probability * 100}%"></i><b style="left:${p.threshold * 100}%"></b></div>
        <div class="m-foot">${esc(foot)}</div>
        ${p.input_preview ? `<div class="m-prev">“${esc(p.input_preview)}”</div>` : ""}
      </div>`;
    });
    const used = new Set(preds.map((p) => p.model));
    for (const [m, desc] of Object.entries(MODEL_INFO)) {
      if (!used.has(m)) {
        const err = (d.errors || []).find((e) => e.model === m);
        cards.push(`<div class="model missing"><div class="m-head"><span class="m-name">${m}</span><span class="m-src">${desc}</span></div>
          <div class="m-foot" style="margin-top:8px">${err ? "Error: " + esc(err.error) : "Not applicable to this input"}</div></div>`);
      }
    }
    $("models").innerHTML = cards.join("");

    // fusion contributions (diverging bar around 0)
    const f = d.fusion || {};
    $("fmethod").textContent = f.method
      ? `${f.method}. Fused p = ${pct(f.probability)}.` + (f.strategy_note ? ` ${f.strategy_note}` : "")
      : "";
    const wl = f.strategy === "gated" ? "gate" : "wt";
    const maxAbs = Math.max(1, ...(f.contributions || []).map((c) => Math.abs(c.logit_contribution)));
    $("contrib").innerHTML = (f.contributions || []).map((c) => {
      const m = f.modalities[c.modality];
      const w = (Math.abs(c.logit_contribution) / maxAbs) * 50;
      const pos = c.logit_contribution >= 0;
      return `<div class="contrib-row" title="weight share ${(c.weight_share * 100).toFixed(0)}%, p=${m.probability}">
        <span>${esc(c.modality)}</span>
        <span class="track"><i class="${pos ? "s" : "n"}" style="${pos ? "left:50%" : `left:${50 - w}%`};width:${w}%"></i></span>
        <span class="muted small">${(c.weight_share * 100).toFixed(0)}% ${wl}</span></div>`;
    }).join("") + (f.contributions?.length ? `<div class="legend"><span>← not sarcastic</span><span>sarcastic →</span></div>` : `<p class="muted">No signals.</p>`);

    // flags
    $("flags").innerHTML = (r.flags || []).length
      ? r.flags.map((x) => `<li><span class="sev ${esc(x.severity)}">${esc(x.severity)}</span>${esc(x.message)}</li>`).join("")
      : `<li>No issues detected. All available channels were analysed cleanly.</li>`;

    // extracted content tabs
    const inp = d.input || {};
    const tabs = [];
    const seg = (src) => (inp.text_segments || []).find((s) => s.source === src);
    if (seg("typed")) tabs.push(["Typed text", esc(seg("typed").text)]);
    if (inp.transcript) {
      const t = inp.transcript;
      const body = (t.segments || []).length
        ? t.segments.map((s) => `<div class="seg"><span class="ts">${s.start.toFixed(1)}–${s.end.toFixed(1)}s</span><span>${esc(s.text)}</span></div>`).join("")
        : `<span class="muted">No speech detected.</span>`;
      tabs.push([`Transcript${t.language ? " (" + t.language + ")" : ""}`, body]);
    }
    if (inp.ocr) {
      const body = inp.ocr.lines.length
        ? inp.ocr.frames.map((fr) => `<div class="seg"><span class="ts">@ ${fr.t.toFixed(1)}s</span><span>${fr.lines.map(esc).join("<br>") || '<span class="muted">—</span>'}</span></div>`).join("")
        : `<span class="muted">No on-screen text found.</span>`;
      tabs.push(["On-screen text (OCR)", body]);
    }
    if (seg("document")) tabs.push([`Document text${inp.document?.pages ? ` (${inp.document.pages} pages)` : ""}`, esc(seg("document").text)]);
    if (inp.media && Object.keys(inp.media).length) tabs.push(["Media info", esc(JSON.stringify(inp.media, null, 2))]);
    if (!tabs.length) tabs.push(["—", `<span class="muted">Nothing extracted.</span>`]);
    $("tabs").innerHTML = tabs.map((t, i) => `<button class="tab" role="tab" aria-selected="${i === 0}" data-i="${i}">${t[0]}</button>`).join("");
    const show = (i) => {
      $("tabpanel").innerHTML = tabs[i][1];
      document.querySelectorAll(".tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.i == i));
    };
    document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => show(+b.dataset.i)));
    show(0);

    $("timings").innerHTML = Object.entries(d.timings_s || {}).map(([k, v]) => `<span>${esc(k.replace(/_/g, " "))}: <b>${v}s</b></span>`).join("");
    $("json").textContent = JSON.stringify(d, null, 2);
    $("res").hidden = false;
  }
})();
