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
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
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
    // No source tag: single receiver, nothing to distinguish.
    body += '<div class="card-head"><span class="ts">' + fmtTs(m.ts_start_ms) +
      '</span><span class="dur">' + fmtDur(m.duration_ms) + "</span></div>";
    // Alert banners (Slice 2): emergency wins when both bits are set.
    // Icons match the approved T10 mockup. Bit 4 is the Slice-3 disaster
    // cue hit: same triangle, violet, so event traffic reads distinct.
    if (m.alert & 4) {
      body += '<div class="banner disaster">' +
        '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">' +
        '<path d="M7 1L13.5 12.5H0.5Z" fill="none" stroke="currentColor"' +
        ' stroke-width="1.8" stroke-linejoin="round"/>' +
        '<line x1="7" y1="5.5" x2="7" y2="9" stroke="currentColor" stroke-width="1.8"/>' +
        '<circle cx="7" cy="10.8" r="1" fill="currentColor"/></svg>' +
        "<span>DISASTER</span></div>";
    } else if (m.alert & 2) {
      body += '<div class="banner emergency">' +
        '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">' +
        '<path d="M7 1L13.5 12.5H0.5Z" fill="none" stroke="currentColor"' +
        ' stroke-width="1.8" stroke-linejoin="round"/>' +
        '<line x1="7" y1="5.5" x2="7" y2="9" stroke="currentColor" stroke-width="1.8"/>' +
        '<circle cx="7" cy="10.8" r="1" fill="currentColor"/></svg>' +
        "<span>EMERGENCY</span></div>";
    } else if (m.alert & 1) {
      body += '<div class="banner foryou">' +
        '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">' +
        '<circle cx="7" cy="4.5" r="2.5" fill="currentColor"/>' +
        '<path d="M1.5 13c0-3 2.5-4.5 5.5-4.5S12.5 10 12.5 13" fill="none"' +
        ' stroke="currentColor" stroke-width="1.8"/></svg>' +
        "<span>FOR YOU</span></div>";
    }
    if (m.status === "failed") {
      body += '<div class="err">' +
        esc(m.fail_reason || "transcription failed — clip kept") + "</div>";
    } else if (!m.corrected_text) {
      body += '<p class="transcript">' + esc(m.transcript) + "</p>";
    }
    if (m.corrected_text) {
      body += '<p class="transcript" data-fixed="' + esc(m.corrected_text) +
        '" data-orig="' + esc(m.transcript) + '">' + esc(m.corrected_text) + "</p>" +
        '<div class="correction-row"><span class="badge corrected">corrigé</span>' +
        '<button class="toggle-orig">voir l\u2019original</button></div>';
    }
    if (m.audio_url) {
      body += '<div class="audio-row"><audio controls preload="none" src="' +
        esc(m.audio_url) + '"></audio>' +
        '<button class="speed">1x</button></div><div class="wave"></div>';
    }
    body += '<div class="foot"><span class="badge lang">' + esc(m.lang) +
      '</span><span class="badge ' + conf + '">' + confText + "</span>";
    if (m.noise) {
      body += '<span class="badge noise" title="Hidden when hide-noise is on">' +
        "noise · " + esc(m.noise) + "</span>";
    }
    if (m.sender_callsign) {
      var who = esc(m.sender_callsign) +
        (m.sender_name ? " · " + esc(m.sender_name) : "");
      body += '<span class="badge sender" title="' + esc(m.sender_source || "") +
        '">' + who + "</span>";
    }
    body += '<span class="triage"><button class="edit-correction">corriger</button>';
    if (m.sender_callsign && m.sender_source === "suggested") {
      body += '<button class="confirm-sender">confirmer</button>';
    }
    if (m.sender_callsign) {
      body += '<button class="fix-sender">indicatif</button>';
    }
    body += "</span>";
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
    var toggle = el.querySelector(".toggle-orig");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var p = el.querySelector("p.transcript");
        var showingFixed = p.textContent === p.dataset.fixed;
        p.textContent = showingFixed ? p.dataset.orig : p.dataset.fixed;
        toggle.textContent = showingFixed ? "voir la correction" : "voir l\u2019original";
      });
    }
    el.querySelectorAll(".edit-correction").forEach(function (btn) {
      btn.addEventListener("click", function () {
        openEditor(el, m);
      });
    });
    el.querySelectorAll(".confirm-sender").forEach(function (btn) {
      btn.addEventListener("click", async function () {
        var res = await fetch("/api/messages/" + encodeURIComponent(m.id) + "/confirm-sender", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ callsign: m.sender_callsign }),
        });
        if (!res.ok) return;
        upsert(await res.json());
      });
    });
    el.querySelectorAll(".fix-sender").forEach(function (btn) {
      btn.addEventListener("click", function () {
        openSenderEditor(el, m);
      });
    });
    return el;
  }

  function openSenderEditor(el, m) {
    if (el.querySelector(".sender-edit")) return;
    var box = document.createElement("div");
    box.className = "correction-edit sender-edit";
    box.innerHTML = '<input type="text" aria-label="Callsign">' +
      '<div class="correction-actions"><button class="save">valider</button>' +
      '<button class="cancel">annuler</button>' +
      '<span class="correction-err" style="display:none">échec, réessayez</span></div>';
    var input = box.querySelector("input");
    input.value = m.sender_callsign || "";
    var anchor = el.querySelector("p.transcript") || el.querySelector(".err");
    anchor.after(box);
    input.focus();
    box.querySelector(".cancel").addEventListener("click", function () {
      box.remove();
    });
    box.querySelector(".save").addEventListener("click", async function () {
      var err = box.querySelector(".correction-err");
      err.style.display = "none";
      var res = await fetch("/api/messages/" + encodeURIComponent(m.id) + "/confirm-sender", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ callsign: input.value }),
      });
      if (!res.ok) {
        err.style.display = "";
        return;
      }
      upsert(await res.json());
    });
  }

  function openEditor(el, m) {
    if (el.querySelector(".correction-edit")) return;
    var box = document.createElement("div");
    box.className = "correction-edit";
    box.innerHTML = '<textarea rows="3"></textarea>' +
      '<div class="correction-actions"><button class="save">enregistrer</button>' +
      '<button class="cancel">annuler</button>' +
      '<span class="correction-err" style="display:none">échec, réessayez</span></div>';
    var area = box.querySelector("textarea");
    area.value = m.corrected_text || m.transcript || "";
    var anchor = el.querySelector("p.transcript") || el.querySelector(".err");
    anchor.after(box);
    area.focus();
    box.querySelector(".cancel").addEventListener("click", function () {
      box.remove();
    });
    box.querySelector(".save").addEventListener("click", async function () {
      var err = box.querySelector(".correction-err");
      err.style.display = "none";
      var res = await fetch("/api/messages/" + encodeURIComponent(m.id) + "/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: area.value }),
      });
      if (!res.ok) {
        err.style.display = "";
        return;
      }
      upsert(await res.json());
    });
  }

  function upsert(m) {
    var old = feed.querySelector('[data-id="' + CSS.escape(m.id) + '"]');
    var fresh = card(m);
    if (old) old.replaceWith(fresh);
    else feed.prepend(fresh);
  }

  // hide-noise is a view rule, not a search option: every live fetch
  // carries it when the box is checked, so the feed opens filtered.
  function hideNoiseOn() {
    return document.getElementById("hidenoise").checked;
  }

  function liveParams() {
    var params = new URLSearchParams({ limit: "20" });
    if (hideNoiseOn()) params.set("hide_noise", "true");
    return params;
  }

  async function loadMore() {
    var url;
    if (searching) {
      var params = new URLSearchParams({ limit: "20" });
      if (cursor) params.set("cursor", cursor);
      url = "/api/search?" + searchParams(params);
    } else {
      var live = liveParams();
      if (cursor) live.set("cursor", cursor);
      url = "/api/messages?" + live;
    }
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
    var sender = document.getElementById("sender").value.trim();
    if (sender) params.set("sender", sender);
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
  // Toggling the checkbox reloads the current view at once (live feed or
  // running search) — the box always shows what the feed holds.
  document.getElementById("hidenoise").addEventListener("change", function () {
    pendingNew = 0;
    jumpPill.style.display = "none";
    cursor = null;
    feed.innerHTML = "";
    loadMore();
  });
  // Enter in any filter field runs the search: no mouse round-trip.
  ["q", "sender", "from", "to"].forEach(function (id) {
    document.getElementById(id).addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") {
        ev.preventDefault();
        document.getElementById("apply").click();
      }
    });
  });
  moreBtn.addEventListener("click", loadMore);
  pauseBtn.addEventListener("click", function () {
    paused = !paused;
    pauseBtn.setAttribute("aria-pressed", String(paused));
    pauseBtn.textContent = paused ? "Resume feed" : "Pause feed";
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
    var res = await fetch("/api/messages?" + liveParams());
    if (!res.ok) return;
    var page = await res.json();
    feed.innerHTML = "";
    page.messages.forEach(function (m) { feed.appendChild(card(m)); });
    searching = false;
    cursor = page.next_cursor || null;
    moreBtn.style.display = cursor ? "" : "none";
  }

  var profileSel = document.getElementById("profile");
  var modeBanner = document.getElementById("modeBanner");

  async function setProfile(name) {
    var res = await fetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name }),
    });
    refreshProfile();
  }

  async function refreshProfile() {
    var res = await fetch("/api/profile");
    if (!res.ok) return;
    var p = await res.json();
    // Rebuild options from the server list (no innerHTML: profile names
    // stay text, never markup).
    profileSel.innerHTML = "";
    p.profiles.forEach(function (pr) {
      var o = document.createElement("option");
      o.value = pr.name;
      o.textContent = pr.name;
      if (pr.name === p.active) o.selected = true;
      profileSel.appendChild(o);
    });
    if (p.active !== "Normal") {
      modeBanner.innerHTML = "";
      var span = document.createElement("span");
      span.textContent = "DISASTER MODE \u2014 " + p.active;
      var btn = document.createElement("button");
      btn.textContent = "return to Normal";
      btn.addEventListener("click", function () { setProfile("Normal"); });
      modeBanner.appendChild(span);
      modeBanner.appendChild(btn);
      modeBanner.style.display = "";
    } else {
      modeBanner.style.display = "none";
    }
  }

  profileSel.addEventListener("change", function () {
    setProfile(profileSel.value);
  });

  // SDR tuning (Slice 4 + manual tune): preset dropdown plus a
  // frequency (MHz) + demodulation entry. Server owns the preset
  // list and the supported modes; tune posts {freq_hz, mode} and
  // the pipeline retunes live. Visible only when the input kind is
  // sdr; mic setups never see it.
  var channelSel = document.getElementById("channel");
  var channelWrap = document.getElementById("channelWrap");
  var tuneWrap = document.getElementById("tuneWrap");
  var freqInput = document.getElementById("freq");
  var demodSel = document.getElementById("demod");
  var tuneBtn = document.getElementById("tune");
  var tuneErr = document.getElementById("tuneErr");

  async function setChannel(name) {
    var res = await fetch("/api/source/channel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: name }),
    });
    refreshChannel();
  }

  function mhz(f) {
    return (f / 1e6).toFixed(3) + " MHz";
  }

  async function tune() {
    tuneErr.textContent = "";
    var mhzVal = parseFloat(freqInput.value);
    if (!isFinite(mhzVal)) {
      tuneErr.textContent = "enter MHz";
      return;
    }
    var res = await fetch("/api/source/channel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ freq_hz: Math.round(mhzVal * 1e6), mode: demodSel.value }),
    });
    if (!res.ok) tuneErr.textContent = "rejected (" + res.status + ")";
    refreshChannel();
  }

  async function refreshChannel() {
    var res = await fetch("/api/source");
    if (!res.ok) return;
    var s = await res.json();
    if (s.kind !== "sdr" || !s.channels || !s.channels.length) {
      channelWrap.style.display = "none";
      tuneWrap.style.display = "none";
      return;
    }
    channelWrap.style.display = "";
    tuneWrap.style.display = "";
    channelSel.innerHTML = "";
    var matched = false;
    s.channels.forEach(function (ch) {
      var o = document.createElement("option");
      o.value = ch.name;
      o.textContent = ch.name + " " + mhz(ch.freq_hz);
      if (ch.name === s.active) {
        o.selected = true;
        matched = true;
      }
      channelSel.appendChild(o);
    });
    if (!matched) channelSel.selectedIndex = -1;
    demodSel.innerHTML = "";
    (s.modes || ["nbfm"]).forEach(function (m) {
      var o = document.createElement("option");
      o.value = m;
      o.textContent = m;
      if (m === s.mode) o.selected = true;
      demodSel.appendChild(o);
    });
    if (s.freq_hz) freqInput.value = (s.freq_hz / 1e6).toFixed(3);
  }

  channelSel.addEventListener("change", function () {
    setChannel(channelSel.value);
  });
  tuneBtn.addEventListener("click", tune);
  freqInput.addEventListener("keydown", function (ev) {
    if (ev.key === "Enter") tune();
  });
  refreshChannel();

  // Listen to radio (Slice 3, R7): one press streams /api/live into the
  // audio element (the click is the autoplay gesture); a second press pauses
  // and drops src, which releases the server stream. Transient stalls and
  // pipeline restarts reconnect on their own (a few tries, then honest
  // stop); only an explicit second press means "stay stopped".
  var listenBtn = document.getElementById("listen");
  var liveAudio = document.getElementById("liveAudio");
  var listening = false;
  var listenRetries = 0;
  listenBtn.addEventListener("click", function () {
    if (listening) stopListening();
    else startListening();
  });
  function startListening() {
    listening = true;
    listenRetries = 0;
    listenBtn.setAttribute("aria-pressed", "true");
    listenBtn.textContent = "Stop radio";
    playLive();
  }
  function playLive() {
    liveAudio.src = "/api/live";
    var pr = liveAudio.play();
    if (pr && pr.catch) {
      pr.catch(function () { scheduleRetry(); });
    }
  }
  function scheduleRetry() {
    // Generous on purpose: a transcription stall (drain blocks the frame
    // loop) starves the relay for the length of a segment, and a pipeline
    // restart rebinds the socket seconds later. Backoff caps at 8 s; the
    // counter resets on every playing event, so only a truly dead relay
    // (minutes of nothing) gives up.
    if (!listening) return;
    if (listenRetries >= 60) { stopListening(); return; }
    listenRetries++;
    setTimeout(function () {
      if (!listening) return;
      playLive();
    }, Math.min(1500 * listenRetries, 8000));
  }
  function stopListening() {
    listening = false;
    listenRetries = 0;
    listenBtn.setAttribute("aria-pressed", "false");
    listenBtn.textContent = "Listen to radio";
    liveAudio.pause();
    liveAudio.removeAttribute("src");
    liveAudio.load();
  }
  liveAudio.addEventListener("playing", function () {
    listenRetries = 0;
  });
  liveAudio.addEventListener("error", function () {
    if (listening) scheduleRetry();
  });
  liveAudio.addEventListener("ended", function () {
    if (listening) scheduleRetry();
  });

  var es = new EventSource("/api/events");
  es.addEventListener("message", function (ev) {
    var m;
    try { m = JSON.parse(ev.data); } catch (e) { return; }
    // Profile-switch broadcasts carry no card: refresh the banner instead
    // of feeding them to the card renderer (no id, no transcript).
    if (m && m.profile && !m.id) { refreshProfile(); return; }
    // Noise stays out of a hidden feed even when it arrives live.
    if (m.noise && hideNoiseOn()) return;
    if (searching) {
      // A search is a snapshot: live arrivals wait behind the pill instead
      // of mixing into the filtered results.
      pendingNew++;
      jumpPill.textContent = pendingNew + " new — show latest";
      jumpPill.style.display = "";
      return;
    }
    if (paused || window.scrollY > 400) {
      pendingNew++;
      jumpPill.textContent = pendingNew + " new — show latest";
      jumpPill.style.display = "";
      if (!paused) upsert(m);
    } else {
      upsert(m);
    }
  });

  pauseBtn.textContent = "Pause feed";
  loadMore();
  refreshProfile();
})();
