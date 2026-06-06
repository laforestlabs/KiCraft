// Landing-page motion for KiCraft (see web._render_landing + kc_landing.css).
// Two effects, both degrade gracefully and honor prefers-reduced-motion:
//   1. a typewriter that cycles real example briefs in the hero console
//   2. scroll-reveal that fades sections in as they enter the viewport
// Reads window.KICRAFT_PROMPTS (injected by the page, same global the app uses).
//
// NiceGUI injects the page's ui.html content client-side, so the landing markup
// may not exist yet at DOMContentLoaded. We poll for it (whenReady) before wiring
// anything up, rather than running once and finding nothing.
(function () {
  "use strict";
  var reduce = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function typewriter() {
    var el = document.querySelector(".kc-type");
    if (!el) return;
    var prompts = (window.KICRAFT_PROMPTS || []).filter(Boolean);
    if (!prompts.length) { return; }            // server seeds a first line already
    if (reduce) { el.textContent = prompts[0]; return; }

    // Start from whatever is currently shown (the server seed), then delete it and
    // cycle onward, so there is no empty flash and no double-typing of line one.
    var cur = (el.textContent || "").trim();
    var i = prompts.indexOf(cur);
    if (i < 0) { i = 0; }
    var j = prompts[i].length;
    var deleting = true;

    function tick() {
      var full = prompts[i];
      el.textContent = full.slice(0, j);
      var delay;
      if (!deleting) {
        j++;
        delay = 38 + Math.random() * 34;
        if (j > full.length) { deleting = true; delay = 1700; }   // hold, then erase
      } else {
        j--;
        delay = 18;
        if (j <= 0) { j = 0; deleting = false; i = (i + 1) % prompts.length; delay = 320; }
      }
      setTimeout(tick, delay);
    }
    tick();
  }

  function reveal() {
    var items = document.querySelectorAll(".kc-reveal");
    if (!items.length) return;
    if (reduce || !("IntersectionObserver" in window)) {
      items.forEach(function (n) { n.classList.add("kc-in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("kc-in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -10% 0px", threshold: 0.08 });
    items.forEach(function (n) { io.observe(n); });
  }

  // Wait for the (client-injected) landing markup, then run once.
  function whenReady(cb, tries) {
    tries = tries || 0;
    if (document.querySelector(".kc-type") || tries > 120) { cb(); return; }
    setTimeout(function () { whenReady(cb, tries + 1); }, 100);
  }

  function start() { whenReady(function () { typewriter(); reveal(); }); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
