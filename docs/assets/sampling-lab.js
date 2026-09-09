// Expected label counts for an illustrative mixture; no spectra or model scores.
(() => {
  "use strict";
  const alphaInput = document.getElementById("sampling-alpha");
  const bandInput = document.getElementById("sampling-band");
  const countInput = document.getElementById("sampling-count");
  const svg = document.getElementById("sampling-chart");
  if (!alphaInput || !bandInput || !countInput || !svg) return;

  const ns = "http://www.w3.org/2000/svg";
  const format = value => value.toLocaleString("en-US", { maximumFractionDigits: 1 });
  function element(tag, attributes, text) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function render() {
    const alpha = Number(alphaInput.value) / 100;
    const band = Number(bandInput.value) / 100;
    const count = Number(countInput.value);
    const uniformInside = count * band / 0.6;
    const focusedInside = count * (alpha + (1 - alpha) * band / 0.6);
    document.getElementById("sampling-alpha-value").textContent = `${alphaInput.value}%`;
    document.getElementById("sampling-band-value").textContent = `${bandInput.value}%`;
    document.getElementById("sampling-count-value").textContent = format(count);
    document.getElementById("sampling-uniform").textContent = format(uniformInside);
    document.getElementById("sampling-focused").textContent = format(focusedInside);
    document.getElementById("sampling-outside").textContent = format(count - focusedInside);
    document.getElementById("sampling-explanation").textContent =
      `For ${format(count)} training rows, ${alphaInput.value}% extra allocation within ±${bandInput.value}% gives ${format(focusedInside)} expected rows there, compared with ${format(uniformInside)} under uniform sampling. At the same total size, coverage outside that band changes from ${format(count - uniformInside)} to ${format(count - focusedInside)}.`;

    // Integrate the two uniform densities over equal-width P bins. This also
    // handles a focus boundary falling inside a bin, without rounding labels.
    const bins = Array.from({ length: 24 }, (_, i) => {
      const left = -0.6 + i * 0.05;
      const right = left + 0.05;
      const overlap = Math.max(0, Math.min(right, band) - Math.max(left, -band));
      return count * ((1 - alpha) / 24 + alpha * overlap / (2 * band));
    });
    const uniform = count / 24;
    const maximum = Math.max(uniform, ...bins) * 1.1;
    const canvasWidth = Math.max(280, Math.min(640, svg.clientWidth || 640));
    const x0 = 58, y0 = 234, width = canvasWidth - 86, height = 190;
    svg.setAttribute("viewBox", `0 0 ${canvasWidth} 290`);
    svg.replaceChildren();
    svg.append(element("title", {}, "Expected counts across the polarization range"));
    svg.append(element("desc", {}, `A mixture of broad and low-polarization uniform draws, totaling ${count} expected training rows. Bar widths are 5 percentage points in P.`));
    for (let tick = 0; tick <= 4; tick++) {
      const y = y0 - tick * height / 4;
      svg.append(element("line", { x1: x0, y1: y, x2: x0 + width, y2: y, stroke: "#d5dcd2" }));
      svg.append(element("text", { x: x0 - 8, y: y + 4, "text-anchor": "end" }, format(maximum * tick / 4)));
    }
    for (let i = 0; i < bins.length; i++) {
      const x = x0 + i * width / bins.length + 2;
      const barWidth = width / bins.length - 4;
      const y = y0 - height * bins[i] / maximum;
      const bar = element("rect", { x, y, width: barWidth, height: y0 - y, fill: "#267b71", "data-expected-count": bins[i] });
      bar.append(element("title", {}, `${-60 + i * 5}% to ${-55 + i * 5}% P: ${format(bins[i])} expected rows`));
      svg.append(bar);
      const uniformY = y0 - height * uniform / maximum;
      svg.append(element("rect", { x, y: uniformY, width: barWidth, height: y0 - uniformY, fill: "none", stroke: "#b44c30", "stroke-width": 1.5 }));
    }
    for (const percent of [-60, -30, 0, 30, 60]) {
      svg.append(element("text", { x: x0 + (percent + 60) / 120 * width, y: y0 + 20, "text-anchor": "middle" }, `${percent}%`));
    }
    svg.append(element("text", { x: x0, y: 24 }, "Expected training rows per bin"));
    svg.append(element("text", { x: x0 + width / 2, y: 280, "text-anchor": "middle" }, "Signed polarization P"));
  }

  for (const input of [alphaInput, bandInput, countInput]) {
    input.disabled = false;
    input.addEventListener("input", render);
  }
  document.getElementById("sampling-chart-container").hidden = false;
  window.addEventListener("resize", render);
  render();
})();
