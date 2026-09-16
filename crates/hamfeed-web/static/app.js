// hamfeed live feed: REST history + SSE append + triage (T11).
// Same card structure as the approved T10 mockup; data comes from /api/*.
(function () {
  "use strict";
  var feed = document.getElementById("feed");
  var moreBtn = document.getElementById("more");
  var jumpPill = document.getElementById("jumpLive");
  var pauseBtn = document.getElementById("pause");
  var cursor = null;
  var paused = false;
  var pendingNew = 0;
  var searching = false;

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtTs(ms) {
    var d = new Date(ms);
    function p(n, w) { return String(n).padStart(w || 2, "0"); }
    return p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  }

  function fmtDur(ms) {
    if (ms == null) return "--:--";
    var s = Math.round(ms / 1000);
    return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0");
  }

  function waveSeed(id) {
    var h = 0;
    for (var i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
    return h;
  }

  function card(m) {
    var el = document.createElement("article");
    el.className = "card";
    el.dataset.id = m.id;
    var conf = m.status === "failed" ? "failed" : m.conf_flag;
    var confText = m.status === "failed"
      ? "failed"
      : esc(m.conf_flag) + " · " + Number(m.stt_conf).toFixed(2);
    var body = "";
    body += '<div class="card-head"><span class="ts">' + fmtTs(m.ts_start_ms) +
      "</span><span class=\"freq\">" + esc(m.freq_label) +
      '</span><span class="dur">' + fmtDur(m.duration_ms) + "</span></div>";
    if (m.status === "failed") {
      body += '<div class="err">' +
        esc(m.fail_reason || "transcription failed — clip kept") + "</div>";
    } else {
      body += '<p class="transcript">' + esc(m.transcript) + "</p>";
    }
    if (m.audio_url) {
      body += '<div class="audio-row"><audio controls preload="none" src="' +
        esc(m.audio_url) + '"></audio>' +
        '<button class="speed">1x</button></div><div class="wave"></div>';
    }
    body += '<div class="foot"><span class="badge lang">' + esc(m.lang) +
      '</span><span class="badge ' + conf + '">' + confText + "</span>";
    if (m.status === "failed") {
      body += '<span class="triage">' +
        '<button data-act="keep">Keep</button>' +
        '<button data-act="drop">Drop</button>' +
        '<button data-act="retry">Retry</button>' +
        '<button data-act="flag">Flag</button></span>';
    }
    body += "</div>";
    el.innerHTML = body;
    // Decorative waveform, seeded per message.
    var wave = el.querySelector(".wave");
    if (wave) {
      var x = waveSeed(m.id);
      for (var i = 0; i < 48; i++) {
        x = (x * 1103515245 + 12345) & 0x7fffffff;
        var bar = document.createElement("i");
        bar.style.height = 3 + (x % 23) + "px";
        wave.appendChild(bar);
      }
    }
    var speed = el.querySelector(".speed");
    if (speed) {
      speed.addEventListener("click", function () {
        var audio = el.querySelector("audio");
        var slow = speed.textContent !== "0.75x";
        speed.textContent = slow ? "0.75x" : "1x";
        audio.playbackRate = slow ? 0.75 : 1.0;
      });
    }
    el.querySelectorAll("[data-act]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        triage(m.id, btn.dataset.act);
      });
    });
    return el;
  }

  function upsert(m) {
    var old = feed.querySelector('[data-id="' + CSS.escape(m.id) + '"]');
    var fresh = card(m);
    if (old) old.replaceWith(fresh);
    else feed.prepend(fresh);
  }

  async function loadMore() {
    var params = new URLSearchParams({ limit: "20" });
    if (cursor) params.set("cursor", cursor);
    var url = searching ? "/api/search?" + searchParams(params) : "/api/messages?" + params;
    var res = await fetch(url);
    if (!res.ok) return;
    var page = await res.json();
    page.messages.forEach(function (m) { feed.appendChild(card(m)); });
    cursor = page.next_cursor || null;
    moreBtn.style.display = cursor ? "" : "none";
    if (!page.messages.length && !feed.children.length) {
      feed.innerHTML = '<div class="empty">No messages yet — leave the receiver running.</div>';
    }
  }

  function searchParams(params) {
    params = params || new URLSearchParams();
    var q = document.getElementById("q").value.trim();
    if (q) params.set("q", q);
    var from = document.getElementById("from").value;
    var to = document.getElementById("to").value;
    if (from) params.set("from", String(new Date(from).getTime()));
    if (to) params.set("to", String(new Date(to + "T23:59:59").getTime()));
    if (document.getElementById("hidenoise").checked) params.set("hide_noise", "true");
    params.set("limit", "20");
    return params.toString();
  }

  async function triage(id, act) {
    var res = await fetch("/api/messages/" + encodeURIComponent(id) + "/" + act, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (!res.ok) return;
    upsert(await res.json());
  }

  document.getElementById("apply").addEventListener("click", function () {
    searching = true;
    cursor = null;
    feed.innerHTML = "";
    loadMore();
  });
  moreBtn.addEventListener("click", loadMore);
  pauseBtn.addEventListener("click", function () {
    paused = !paused;
    pauseBtn.setAttribute("aria-pressed", String(paused));
    pauseBtn.textContent = paused ? "resume live" : "pause live";
    if (!paused && pendingNew > 0) {
      pendingNew = 0;
      jumpPill.style.display = "none";
      refreshNewest();
    }
  });
  jumpPill.addEventListener("click", function () {
    pendingNew = 0;
    jumpPill.style.display = "none";
    refreshNewest();
  });

  async function refreshNewest() {
    var res = await fetch("/api/messages?limit=20");
    if (!res.ok) return;
    var page = await res.json();
    feed.innerHTML = "";
    page.messages.forEach(function (m) { feed.appendChild(card(m)); });
    searching = false;
    cursor = page.next_cursor || null;
    moreBtn.style.display = cursor ? "" : "none";
  }

  var es = new EventSource("/api/events");
  es.addEventListener("message", function (ev) {
    var m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (paused || window.scrollY > 400) {
      pendingNew++;
      jumpPill.textContent = pendingNew + " new — jump to live";
      jumpPill.style.display = "";
      if (!paused) upsert(m);
    } else {
      upsert(m);
    }
  });

  loadMore();
})();
