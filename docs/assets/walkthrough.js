"use strict";

document.querySelectorAll("pre[data-run]").forEach((pre, index) => {
  const code = pre.querySelector("code");
  const button = document.createElement("button");
  button.className = "copy-button";
  button.type = "button";
  button.textContent = "Copy";
  button.setAttribute("aria-label", `Copy command block ${index + 1}`);
  button.addEventListener("click", async () => {
    const status = document.getElementById("copy-status");
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = "Copied";
      status.textContent = `Command block ${index + 1} copied.`;
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      button.textContent = "Selected";
      status.textContent = "Commands selected. Use your keyboard’s copy command.";
    }
    window.setTimeout(() => { button.textContent = "Copy"; }, 2200);
  });
  pre.appendChild(button);
});
