"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const green = "#267b71", rust = "#b44c30", grid = "#dce2d8";
  const signed = (n, digits = 3) => `${n >= 0 ? "+" : "−"}${Math.abs(n).toFixed(digits)}`;
  const escape = (value) => String(value).replace(/[&<>"']/g, c => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));

  function axes(svg, xMin, xMax, yMin, yMax, xLabel, formatX, formatY, logarithmic = false) {
    const [, , width, height] = svg.getAttribute("viewBox").split(" ").map(Number);
    const left = 62, right = width - 17, top = 28, bottom = height - 43;
    const x = value => left + (value - xMin) / (xMax - xMin) * (right - left);
    const transform = value => logarithmic ? Math.log10(value) : value;
    const y = value => bottom - (transform(value) - transform(yMin)) / (transform(yMax) - transform(yMin)) * (bottom - top);
    let markup = "";
    for (let i = 0; i <= 4; i++) {
      const t = i / 4;
      const vx = xMin + t * (xMax - xMin);
      const vy = logarithmic ? 10 ** (Math.log10(yMin) + t * Math.log10(yMax / yMin)) : yMin + t * (yMax - yMin);
      markup += `<line x1="${left}" y1="${y(vy)}" x2="${right}" y2="${y(vy)}" stroke="${grid}"/><text x="${left - 9}" y="${y(vy) + 4}" text-anchor="end">${escape(formatY(vy))}</text>`;
      markup += `<text x="${x(vx)}" y="${bottom + 20}" text-anchor="middle">${escape(formatX(vx))}</text>`;
    }
    markup += `<text x="${(left + right) / 2}" y="${height - 3}" text-anchor="middle">${escape(xLabel)}</text>`;
    return {x, y, left, right, top, bottom, markup};
  }

  const path = (points, x, y) => points.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(2)},${y(b).toFixed(2)}`).join(" ");

  function updateLab() {
    const bias = Number($("bias").value), width = Number($("width").value), p0 = Number($("p0").value);
    const rmse = Math.hypot(bias, width);
    $("bias-value").textContent = `${signed(bias)} pp`;
    $("width-value").textContent = `${width.toFixed(3)} pp`;
    $("p0-value").textContent = `${p0.toFixed(1)}%`;
    $("lab-rmse").textContent = `${rmse.toFixed(4)} pp`;
    $("lab-relative-bias").textContent = p0 === 0 ? "Undefined" : `${signed(100 * bias / p0)}%`;
    $("lab-relative-width").textContent = p0 === 0 ? "Undefined" : `${(100 * width / p0).toFixed(3)}%`;
    $("lab-relative-rmse").textContent = p0 === 0 ? "Undefined" : `${(100 * rmse / p0).toFixed(3)}%`;
    const pass = rmse <= 0.05;
    $("gate-status").textContent = `${pass ? "Within" : "Above"} the 0.05 pp RMSE target for this hypothetical distribution.${p0 === 0 ? " Relative error is undefined at zero polarization." : ""}`;
    $("gate-status").classList.toggle("pass", pass);
    const svg = $("error-chart");
    const density = value => Math.exp(-0.5 * ((value - bias) / width) ** 2) / (width * Math.sqrt(2 * Math.PI));
    const a = axes(svg, -0.65, 0.65, 0, density(bias) * 1.15, "Residual (percentage points)", v => v.toFixed(2), v => v.toFixed(v < 10 ? 1 : 0));
    const points = Array.from({length: 601}, (_, i) => { const e = -0.65 + i * 1.3 / 600; return [e, density(e)]; });
    const d = path(points, a.x, a.y);
    svg.innerHTML = `<title>Illustrative Gaussian: bias ${bias.toFixed(3)} pp, width ${width.toFixed(3)} pp, RMSE ${rmse.toFixed(4)} pp</title>` + a.markup +
      `<rect x="${a.x(-0.05)}" y="${a.top}" width="${a.x(0.05) - a.x(-0.05)}" height="${a.bottom - a.top}" fill="${green}" opacity=".08"/>
       <path d="${d} L${a.right},${a.bottom} L${a.left},${a.bottom} Z" fill="${green}" opacity=".12"/>
       <line x1="${a.x(0)}" x2="${a.x(0)}" y1="${a.top}" y2="${a.bottom}" stroke="#76887d" stroke-dasharray="3 4"/>
       <path d="${d}" fill="none" stroke="${green}" stroke-width="2.5"/>
       <line x1="${a.x(bias)}" x2="${a.x(bias)}" y1="${a.top}" y2="${a.bottom}" stroke="${rust}" stroke-dasharray="5 3"/>
       <text x="${a.left}" y="13">Density (per pp)</text>`;
  }

  ["bias", "width", "p0"].forEach(id => $(id).addEventListener("input", updateLab));
  $("reset-lab").addEventListener("click", () => {
    $("bias").value = "0.02"; $("width").value = "0.05"; $("p0").value = "5"; updateLab();
  });
  updateLab();

  function renderSpectrum() {
    const s = window.TUTORIAL_RESULTS?.spectrum;
    if (!s) return;
    const mode = $("spectrum-select").value;
    const values = mode === "raw" ? s.signal : mode === "noise" ? s.noise : s.lineshape.map((v, i) => v + s.noise[i]);
    const reference = mode === "raw" ? s.baseline : mode === "residual" ? s.lineshape : null;
    const all = reference ? values.concat(reference) : values;
    const low = Math.min(...all), high = Math.max(...all), pad = (high - low) * 0.13;
    const svg = $("spectrum-chart");
    const digits = Math.max(0, Math.min(9, Math.ceil(-Math.log10((high - low) / 4)) + 1));
    const formatVoltage = v => high - low < Math.max(Math.abs(low), Math.abs(high)) * 0.1 ? v.toFixed(digits) : v.toExponential(2);
    const a = axes(svg, s.frequency[0], s.frequency.at(-1), low - pad, high + pad,
                   "Frequency (MHz)", v => v.toFixed(1), formatVoltage);
    svg.innerHTML = `<title>${escape(s.label)}; ${escape(mode)} view</title>` + a.markup +
      `<path d="${path(s.frequency.map((f, i) => [f, values[i]]), a.x, a.y)}" fill="none" stroke="${green}" stroke-width="1.8"/>` +
      (reference ? `<path d="${path(s.frequency.map((f, i) => [f, reference[i]]), a.x, a.y)}" fill="none" stroke="${rust}" stroke-width="1.5" stroke-dasharray="6 4"/>` : "") +
      `<text x="${a.left}" y="13">Detector voltage (V)</text>`;
    const explanation = mode === "raw" ? "Green: full circuit sweep. Dashed rust: physical Q-curve at zero susceptibility." : mode === "residual" ? "Green: circuit-coupled Pake doublet + noise. Dashed rust: clean signal, V(χ) − V(0)." : "White-Gaussian detector-noise reference scenario, not a fitted experimental covariance.";
    const c = s.configuration;
    $("spectrum-caption").textContent = `${explanation} P = ${(100*s.p).toFixed(0)}%, g = ${c.g.toFixed(3)}, η = ${c.eta.toFixed(3)}, noise SD ${c.noise_rms_v.toExponential(1)} V. Vertical scales change between views.`;
  }
  $("spectrum-select").addEventListener("change", renderSpectrum);
  renderSpectrum();

  function renderRun(run) {
    const history = run.history;
    const losses = history.flatMap(row => [row.train_loss, row.val_loss]);
    const best = history.reduce((a, b) => b.val_loss < a.val_loss ? b : a);
    const svg = $("history-chart");
    const a = axes(svg, 1, Math.max(2, history.at(-1).epoch), Math.min(...losses) * 0.8, Math.max(...losses) * 1.2,
                   "Epoch", v => String(Math.round(v)), v => v.toExponential(1), true);
    svg.innerHTML = `<title>Saved losses for ${escape(run.label)}; best validation epoch ${best.epoch}</title>` + a.markup +
      `<path d="${path(history.map(r => [r.epoch, r.train_loss]), a.x, a.y)}" fill="none" stroke="${green}" stroke-width="2"/>
       <path d="${path(history.map(r => [r.epoch, r.val_loss]), a.x, a.y)}" fill="none" stroke="${rust}" stroke-width="2" stroke-dasharray="5 3"/>
       <circle cx="${a.x(best.epoch)}" cy="${a.y(best.val_loss)}" r="4" fill="${rust}"/>
       <text x="${a.left}" y="13">Normalized MSE · log scale</text>`;
    $("history-caption").textContent = `Green: training. Dashed rust: validation. Best epoch ${best.epoch}; validation RMSE ${(Math.sqrt(best.val_loss) * run.target_scale).toFixed(6)} fractional P. ${run.config.architecture === "physics_multiscale" ? "Training includes dropout." : "The compact model does not use dropout."}`;

    const hist = run.histogram;
    const hsvg = $("residual-chart");
    const b = axes(hsvg, hist.edges[0] * 100, hist.edges.at(-1) * 100, 0, Math.max(...hist.counts) * 1.12,
                   "Residual (percentage points)", v => v.toPrecision(3), v => String(Math.round(v)));
    const bars = hist.counts.map((count, i) => `<rect x="${b.x(100 * hist.edges[i])}" y="${b.y(count)}" width="${Math.max(0, b.x(100 * hist.edges[i + 1]) - b.x(100 * hist.edges[i]) - 0.6)}" height="${b.bottom - b.y(count)}" fill="${green}" opacity=".8"/>`).join("");
    hsvg.innerHTML = `<title>Held-out residual histogram for ${escape(run.label)}</title>` + b.markup + bars +
      `<line x1="${b.x(0)}" x2="${b.x(0)}" y1="${b.top}" y2="${b.bottom}" stroke="${rust}" stroke-dasharray="4 3"/>
       <text x="${b.left}" y="13">Event count</text>`;
    $("residual-caption").textContent = `${run.metrics.n.toLocaleString()} held-out events; ${hist.outside} outside the displayed ±${(100 * hist.edges.at(-1)).toFixed(2)} pp range. Axes change between runs; all events enter the metrics. Absolute-error 95th percentile: ${(100 * run.metrics.p95_absolute_error).toFixed(4)} pp.`;
    const m = run.near_five_percent;
    if (m.n === 0) {
      $("local-result").textContent = "No held-out events in the signed 4.5%–5.5% band. Generate more independent examples before reporting local performance.";
      return;
    }
    $("local-result").innerHTML = `<p><strong>Actually near +5% polarization:</strong> ${m.n.toLocaleString()} held-out events with 4.5% ≤ P &lt; 5.5%.</p><p>Bias <strong>${signed(m.bias * 100, 4)} pp</strong> · width <strong>${(100 * m.width).toFixed(4)} pp</strong> · RMSE <strong>${(100 * m.rmse).toFixed(4)} pp</strong>.<br>Relative to P₀ = 5%: bias ${signed(m.relative_bias_percent_at_p0)}%, width ${m.relative_width_percent_at_p0.toFixed(3)}%, RMSE ${m.relative_rmse_percent_at_p0.toFixed(3)}%. These are descriptive local-band metrics, without uncertainty intervals.</p>`;
  }

  if (window.TUTORIAL_RESULTS?.runs?.length) {
    const runs = window.TUTORIAL_RESULTS.runs;
    $("results-table").innerHTML = runs.map(run => `<tr><td>${escape(run.label)}</td><td>${run.metrics.n.toLocaleString()}</td><td>${signed(run.metrics.bias, 7)}</td><td>${run.metrics.width.toFixed(7)}</td><td>${run.metrics.rmse.toFixed(7)}</td></tr>`).join("");
    $("run-select").addEventListener("change", () => renderRun(runs.find(r => r.id === $("run-select").value)));
    renderRun(runs[0]);
  } else {
    $("local-result").textContent = "The result snapshot could not be loaded. The summary table above remains available.";
  }

  document.querySelectorAll("pre").forEach((pre, index) => {
    const code = pre.querySelector("code");
    const button = document.createElement("button");
    button.className = "copy-button"; button.type = "button"; button.textContent = "Copy";
    button.setAttribute("aria-label", `Copy code example ${index + 1}`);
    button.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(code.textContent);
        button.textContent = "Copied";
        $("copy-status").textContent = `Code example ${index + 1} copied.`;
      } catch {
        const range = document.createRange(); range.selectNodeContents(code);
        const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
        button.textContent = "Selected";
        $("copy-status").textContent = "Code selected. Use your keyboard's copy command.";
      }
      window.setTimeout(() => { button.textContent = "Copy"; }, 2200);
    });
    pre.appendChild(button);
  });

  const links = [...document.querySelectorAll('.sidebar nav a[href^="#"]')];
  const sections = links.map(link => document.querySelector(link.getAttribute("href")));
  let scheduled = false;
  function updateNav() {
    let active = sections[0];
    for (const section of sections) {
      if (section.getBoundingClientRect().top <= 160) active = section;
    }
    links.forEach(link => {
      const selected = link.getAttribute("href") === `#${active.id}`;
      link.classList.toggle("active", selected);
      if (selected) link.setAttribute("aria-current", "location"); else link.removeAttribute("aria-current");
    });
    scheduled = false;
  }
  window.addEventListener("scroll", () => {
    if (!scheduled) { scheduled = true; window.requestAnimationFrame(updateNav); }
  }, {passive: true});
  updateNav();

  let closedDetails = [];
  window.addEventListener("beforeprint", () => {
    closedDetails = [...document.querySelectorAll("details:not([open])")];
    closedDetails.forEach(details => { details.open = true; });
  });
  window.addEventListener("afterprint", () => {
    closedDetails.forEach(details => { details.open = false; });
    closedDetails = [];
  });
  $("print-page").addEventListener("click", () => window.print());
})();
