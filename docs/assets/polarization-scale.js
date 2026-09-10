// Saved model predictions and a separate, explicitly hypothetical 1/|P| lesson.
(() => {
  "use strict";
  const $ = id => document.getElementById(id);
  const colors = ["#b44c30", "#57786e", "#173c3a"];
  const ns = "http://www.w3.org/2000/svg";
  const format = value => value === null ? "—" : Number(value.toPrecision(5)).toString();
  function node(tag, attributes = {}, text) {
    const element = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function axes(svg, values, title, logX = false) {
    const width = Math.max(260, Math.min(760, svg.clientWidth || 760));
    const left = 64, right = width - 18, top = 40, bottom = 282;
    const minimum = Math.min(...values) / 1.6, maximum = Math.max(...values) * 1.6;
    const x = value => left + (logX ? Math.log(value / .1) / Math.log(250) : value / 25) * (right - left);
    const y = value => bottom - Math.log(value / minimum) / Math.log(maximum / minimum) * (bottom - top);
    svg.setAttribute("viewBox", `0 0 ${width} 330`);
    svg.replaceChildren(node("title", {}, title));
    for (let i = 0; i <= 4; i++) {
      const value = minimum * (maximum / minimum) ** (i / 4);
      svg.append(node("line", {x1: left, x2: right, y1: y(value), y2: y(value), stroke: "#dce2d8"}));
      svg.append(node("text", {x: left - 8, y: y(value) + 4, "text-anchor": "end"}, value.toPrecision(2)));
    }
    for (const value of (logX ? [.1, 1, 5, 25] : [0, 5, 10, 15, 20, 25])) {
      svg.append(node("text", {x: x(value), y: bottom + 21, "text-anchor": "middle"}, value));
    }
    svg.append(node("text", {x: left, y: 18}, title));
    svg.append(node("text", {x: (left + right) / 2, y: 326, "text-anchor": "middle"}, logX ? "|P| (%) · log scale" : "|True polarization| (%)"));
    return {x, y, top, bottom};
  }
  function line(points, a, attributes) {
    return node("path", {d: points.map(([x, y], i) => `${i ? "L" : "M"}${a.x(x)},${a.y(y)}`).join(" "), fill: "none", ...attributes});
  }
  function metrics(rows, low, high, sign) {
    const chosen = rows.filter(([p]) => (sign === "positive" ? p >= 0 : p < 0) && Math.abs(p) >= low &&
      (high === .25 ? Math.abs(p) <= high : Math.abs(p) < high));
    const errors = chosen.map(([p, prediction]) => p - prediction);
    const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
    const n = errors.length;
    const ordered = errors.map(Math.abs).sort((a, b) => a - b);
    const q = .95 * (n - 1), lo = Math.floor(q), hi = Math.ceil(q);
    return {
      n, n_groups: new Set(chosen.map(row => row[2])).size,
      bias_pp: n ? 100 * mean(errors) : null,
      rmse_pp: n ? 100 * Math.sqrt(mean(errors.map(e => e * e))) : null,
      relative_rms_percent: n && low > 0 ? 100 * Math.sqrt(mean(chosen.map(([p, prediction]) => ((p - prediction) / p) ** 2))) : null,
      p95_absolute_error_pp: n ? 100 * (ordered[lo] + (q - lo) * (ordered[hi] - ordered[lo])) : null
    };
  }

  const snapshot = window.MODEL_COMPARISON;
  function renderObserved() {
    if (!snapshot?.polarization_profile || !$("polarization-chart")) return;
    const sign = $("polarization-sign").value, key = $("polarization-metric").value;
    const scale = Number($("polarization-minimum").value);
    const low = Math.max(0, (scale - .5) / 100), high = Math.min(.25, (scale + .5) / 100);
    const profile = snapshot.polarization_profile.views[sign];
    const selected = snapshot.runs.map(run => metrics(run.test_predictions, low, high, sign));
    const values = profile.flatMap(run => run.bands.map(band => band[key])).concat(selected.map(band => band[key])).filter(v => v !== null && v > 0);
    const title = key === "rmse_pp" ? "RMSE (pp) · log scale" : "Relative RMS (%) · log scale";
    const svg = $("polarization-chart"), a = axes(svg, values, title);
    svg.append(node("rect", {x: a.x(100 * low), y: a.top, width: a.x(100 * high) - a.x(100 * low), height: a.bottom - a.top, fill: "#267b71", opacity: .09}));
    svg.append(node("line", {x1: a.x(scale), x2: a.x(scale), y1: a.top, y2: a.bottom, stroke: "#76887d", "stroke-dasharray": "3 4"}));
    profile.forEach((run, index) => {
      const bands = run.bands.filter(b => b[key] !== null && b[key] > 0);
      svg.append(line(bands.map(b => [50 * (b.low_inclusive + b.high), b[key]]), a, {stroke: colors[index], "stroke-width": 2}));
      for (const b of bands) {
        const label = `${run.label}: ${100 * b.low_inclusive}–${100 * b.high}% |P|; ${b.n} events, ${b.n_groups} groups; ${format(b[key])} ${key === "rmse_pp" ? "pp" : "%"}`;
        const circle = node("circle", {cx: a.x(50 * (b.low_inclusive + b.high)), cy: a.y(b[key]), r: 3.5, fill: colors[index], tabindex: 0,
          "aria-label": label, "data-value": b[key], "data-low": b.low_inclusive, "data-high": b.high, "data-model": run.architecture});
        circle.append(node("title", {}, label));
        svg.append(circle);
      }
      if (selected[index][key] !== null && selected[index][key] > 0) {
        svg.append(node("circle", {cx: a.x(scale), cy: a.y(selected[index][key]), r: 5, fill: "#fffefa", stroke: colors[index], "stroke-width": 2,
          "data-selected-model": run.architecture, "aria-label": `${run.label}, selected band: ${format(selected[index][key])}`}));
      }
    });
    $("polarization-minimum-value").textContent = `${scale.toFixed(1)}%`;
    $("polarization-band-label").textContent = `Evaluate near ${sign === "negative" ? "−" : "+"}${scale.toFixed(1)}%: ${format(100 * low)}% ≤ |P| ${high === .25 ? "≤" : "<"} ${format(100 * high)}%, ${sign === "negative" ? "P < 0" : "P ≥ 0"}. This finite band is not a fixed-P experiment.`;
    $("polarization-caption").textContent = "Filled dots: saved band metrics; lines guide the eye. Hollow dots: the separately recomputed selected band, shaded vertically. The error axis adjusts to the selected data. " +
      (key === "relative_rms_percent" ? "Relative values use each event's true P. Bands touching zero are omitted from this curve; their absolute errors remain available." : "All selected rows enter absolute RMSE, including bands near zero.");
    const table = $("polarization-band-table");
    table.replaceChildren();
    selected.forEach((m, index) => {
      const row = document.createElement("tr");
      row.dataset.model = snapshot.runs[index].architecture;
      for (const key of ["label", "n", "n_groups", "bias_pp", "rmse_pp", "relative_rms_percent", "p95_absolute_error_pp"]) {
        const td = document.createElement("td");
        td.textContent = key === "label" ? snapshot.runs[index].label : format(m[key]);
        if (key !== "label") { td.dataset.metric = key; td.dataset.value = m[key] === null ? "null" : m[key]; }
        row.append(td);
      }
      table.append(row);
    });
    const count = selected[0];
    $("polarization-count-note").textContent = count.n ?
      `All three models use the same ${count.n} events from ${count.n_groups} of the ${snapshot.group_values.test.length} test configurations in this band. Related rows and overlapping bands are not independent acquisitions. These descriptive estimates have no uncertainty intervals.${low === 0 ? " Relative RMS is omitted because the band touches zero; no denominator floor is used." : ""}` :
      "No saved test events in this band. Metrics are unavailable, not zero error; more independent evaluation data are needed at this scale.";
  }
  if (snapshot?.polarization_profile && $("polarization-live")) {
    $("polarization-live").hidden = false;
    $("polarization-static").open = false;
    for (const id of ["polarization-sign", "polarization-metric", "polarization-minimum"]) {
      $(id).disabled = false;
      $(id).addEventListener(id === "polarization-minimum" ? "input" : "change", renderObserved);
    }
    renderObserved();
    window.addEventListener("resize", renderObserved);
  }

  function renderIdeal() {
    const svg = $("polarization-ideal-chart");
    if (!svg || !$("bias")) return;
    const rmse = Math.hypot(Number($("bias").value), Number($("width").value));
    const scale = Number($("p0").value);
    const relative = p => 100 * rmse / p;
    const a = axes(svg, [relative(.1), relative(25)], "Relative RMS (%) · log scale", true);
    const points = Array.from({length: 201}, (_, i) => { const p = .1 * 250 ** (i / 200); return [p, relative(p)]; });
    svg.append(line(points, a, {stroke: "#267b71", "stroke-width": 2.5}));
    if (scale > 0) {
      svg.append(node("line", {x1: a.x(scale), x2: a.x(scale), y1: a.top, y2: a.bottom, stroke: "#b44c30", "stroke-dasharray": "3 4"}));
      svg.append(node("circle", {cx: a.x(scale), cy: a.y(relative(scale)), r: 5, fill: "#b44c30", "data-relative": relative(scale)}));
    }
    $("polarization-ideal-caption").textContent = `Absolute RMSE stays ${rmse.toFixed(4)} pp at every P in this hypothetical constant-error model. ${scale > 0 ? `At |P| = ${scale.toFixed(1)}%, relative RMS is ${relative(scale).toFixed(3)}%.` : "Relative error is undefined at P = 0; no point is plotted there."} This curve is a unit-scaling calculation, not a measured model result.`;
  }
  if ($("polarization-ideal-chart")) {
    $("polarization-ideal-live").hidden = false;
    for (const id of ["bias", "width", "p0"]) $(id).addEventListener("input", renderIdeal);
    $("reset-lab").addEventListener("click", renderIdeal);
    window.addEventListener("resize", renderIdeal);
    renderIdeal();
  }
})();
