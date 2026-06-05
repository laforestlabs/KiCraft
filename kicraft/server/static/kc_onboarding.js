// Animated cycling placeholder for the KiCraft landing page.
//
// Types each brief from window.KICRAFT_PROMPTS into the prompt textarea's
// placeholder, holds with a blinking cursor, deletes, and advances, looping
// forever. Idles (restoring window.KICRAFT_PLACEHOLDER_FALLBACK) whenever the user
// focuses the box or it has any text, and resumes once it is empty + blurred. Under
// prefers-reduced-motion it rotates whole strings instead of typing per character.
//
// It only ever writes the native `placeholder` attribute, never `value`, so it does
// not fight NiceGUI/Vue (which keeps the placeholder prop constant after mount).
// Also hides a previously-dismissed .kc-welcome card on load (localStorage).
(function () {
  'use strict';

  var CURSOR = '▍';          // ▍ trailing caret
  var TYPE_MS = 45, DELETE_MS = 25, HOLD_MS = 1600, GAP_MS = 450, BLINK_MS = 500;
  var IDLE_RECHECK_MS = 400;

  var reduce = !!(window.matchMedia &&
                  window.matchMedia('(prefers-reduced-motion: reduce)').matches);

  function prompts() {
    var p = window.KICRAFT_PROMPTS;
    return (Array.isArray(p) && p.length) ? p : [];
  }
  function fallback() { return window.KICRAFT_PLACEHOLDER_FALLBACK || ''; }
  function findBox() { return document.querySelector('.kc-brief textarea'); }

  function start(el) {
    var list = prompts();
    if (!list.length) return;
    var idx = 0, timer = null, blink = null;

    function setPh(s) { el.setAttribute('placeholder', s); }
    function schedule(fn, ms) { timer = setTimeout(fn, ms); }
    // Suspended while the user is engaged with the field.
    function suspended() {
      return el.value.length > 0 || document.activeElement === el;
    }
    // Park: show the static fallback and re-poll so we resume the instant the box
    // is empty + blurred again.
    function park(next) {
      setPh(fallback());
      schedule(function () { suspended() ? park(next) : next(); }, IDLE_RECHECK_MS);
    }

    function typeIn() {
      if (suspended()) { park(typeIn); return; }
      var full = list[idx], n = 0;
      (function step() {
        if (suspended()) { park(typeIn); return; }
        n++;
        setPh(full.slice(0, n) + CURSOR);
        n < full.length ? schedule(step, TYPE_MS) : hold(full);
      })();
    }

    function hold(full) {
      var on = true;
      blink = setInterval(function () {
        if (suspended()) { clearInterval(blink); blink = null; return; }
        on = !on;
        setPh(full + (on ? CURSOR : ' '));
      }, BLINK_MS);
      schedule(function () {
        if (blink) { clearInterval(blink); blink = null; }
        deleteOut(full);
      }, HOLD_MS);
    }

    function deleteOut(full) {
      if (suspended()) { park(typeIn); return; }
      var n = full.length;
      (function step() {
        if (suspended()) { park(typeIn); return; }
        n--;
        setPh(full.slice(0, n) + CURSOR);
        if (n > 0) { schedule(step, DELETE_MS); }
        else { idx = (idx + 1) % list.length; schedule(typeIn, GAP_MS); }
      })();
    }

    function rotate() {  // reduced-motion path: swap whole strings
      if (suspended()) { setPh(fallback()); }
      else { setPh(list[idx]); idx = (idx + 1) % list.length; }
      schedule(rotate, 4000);
    }

    reduce ? rotate() : typeIn();
  }

  function dismissWelcome() {
    try {
      if (localStorage.getItem('kc_welcome_dismissed')) {
        var nodes = document.querySelectorAll('.kc-welcome');
        for (var i = 0; i < nodes.length; i++) { nodes[i].style.display = 'none'; }
      }
    } catch (e) { /* private mode / disabled storage: leave the card visible */ }
  }

  function boot() {
    dismissWelcome();
    // NiceGUI mounts elements over the socket after load, so poll for the textarea.
    var tries = 0;
    var iv = setInterval(function () {
      var el = findBox();
      if (el && !el.dataset.kcAnim) {
        el.dataset.kcAnim = '1';
        clearInterval(iv);
        start(el);
      } else if (++tries > 300) {       // ~60s ceiling, then give up quietly
        clearInterval(iv);
      }
    }, 200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
