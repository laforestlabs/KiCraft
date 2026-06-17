// k6 web-tier load driver for kicraft.io.
//
// Hammers the public HTTP routes (landing, login/signup pages, /browse, /samples,
// and -- if TOKEN_URL is given -- a signed project-file fetch). NiceGUI's
// interactive design submit runs over a websocket, NOT these HTTP routes, so this
// script measures the HTTP/SQLite-read surface and the static/landing path; drive
// the full authed submit with scripts/loadtest_web_ws.py (Locust) instead.
//
//   k6 run scripts/loadtest_web.js -e BASE=http://127.0.0.1:8080
//   k6 run -e BASE=https://kicraft.io -e VUS=20 -e DURATION=2m scripts/loadtest_web.js
//   k6 run -e BASE=... -e TOKEN_URL="/project/<token>/A_USB_C.kicad_pcb" scripts/loadtest_web.js
//
// Mint TOKEN_URL out of band with scripts/mint_loadtest_token.py (no prod route).

import http from "k6/http";
import { check, group, sleep } from "k6";

const BASE = __ENV.BASE || "http://127.0.0.1:8080";
const VUS = parseInt(__ENV.VUS || "10");
const DURATION = __ENV.DURATION || "1m";
const TOKEN_URL = __ENV.TOKEN_URL || ""; // e.g. /project/<token>/<file>

export const options = {
  scenarios: {
    ramp: {
      executor: "ramping-vus",
      startVUs: 1,
      stages: [
        { duration: "15s", target: VUS },
        { duration: DURATION, target: VUS },
        { duration: "10s", target: 0 },
      ],
    },
  },
  thresholds: {
    // Tune against the 2-core box: these are starting gates, not guarantees.
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<2000", "p(99)<5000"],
  },
};

const PUBLIC_GETS = ["/", "/login", "/signup", "/browse", "/samples", "/pricing"];

export default function () {
  group("public_pages", () => {
    for (const path of PUBLIC_GETS) {
      const res = http.get(`${BASE}${path}`);
      check(res, { [`${path} 200`]: (r) => r.status === 200 });
    }
  });

  if (TOKEN_URL) {
    group("project_file", () => {
      const res = http.get(`${BASE}${TOKEN_URL}`);
      check(res, { "file served": (r) => r.status === 200 || r.status === 304 });
    });
  }

  sleep(Math.random() * 1.0 + 0.5); // think time
}
