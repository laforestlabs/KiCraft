// Tail-follow autoscroll for the streaming panes (Thinking, Activity / log).
//
// Each `.kc-follow` element is a native `overflow-y:auto` container. It sticks to
// the bottom as content streams in, releases the moment the user scrolls up, and
// re-sticks when the user scrolls back to the bottom.
//
// Why client-side: a native `scroll` event fires only on a real position change,
// never on content growth, so streaming text cannot unstick us. The previous
// server-side approach derived "stick" from Quasar QScrollArea's @scroll, which
// also fires on resize, so the first overflow (still scrolled to the top) wedged
// autoscroll permanently off.
(function () {
  'use strict';
  var THRESHOLD = 24;  // px from the bottom that still counts as "at the bottom"

  function atBottom(el) {
    return el.scrollHeight - el.clientHeight - el.scrollTop <= THRESHOLD;
  }

  function attach(el) {
    if (el.dataset.kcFollow) return;
    el.dataset.kcFollow = '1';
    var stick = true;
    var pending = false;

    el.addEventListener('scroll', function () {
      if (!el.clientHeight) return;   // hidden (inactive tab): ignore
      stick = atBottom(el);           // user scrolls up -> false; back to bottom -> true
    }, { passive: true });

    function pin() {
      pending = false;
      if (stick) { el.scrollTop = el.scrollHeight; }
    }
    // Content growth does not fire a scroll event, so observe mutations and pin
    // to the bottom (one scroll per frame) while stuck.
    new MutationObserver(function () {
      if (stick && !pending) { pending = true; requestAnimationFrame(pin); }
    }).observe(el, { childList: true, subtree: true, characterData: true });

    el.scrollTop = el.scrollHeight;   // start pinned
  }

  function scan() {
    var nodes = document.querySelectorAll('.kc-follow');
    for (var i = 0; i < nodes.length; i++) { attach(nodes[i]); }
  }
  // Panes mount lazily (tab panels) and persist across runs, so a cheap periodic
  // scan wires any that appear later; attach() is idempotent.
  function boot() { scan(); setInterval(scan, 1000); }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else { boot(); }
})();
