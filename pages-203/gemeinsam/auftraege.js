// Aufträge dieses Geräts (Bild, Stimme, Musik über /v1/jobs; Video über /v1/videos), seitenübergreifend.
// Geschrieben mit KI-Unterstützung (Claude Code), 01.11.2026; Vorrang (Priorität je Auftrag) 13.11.2026.
// Jedes Gerät (Browser) bekommt beim ersten Besuch eine zufällige Kennung (localStorage "gufo-geraet"). Lange
// Aufträge laufen als Job auf dem Router weiter, auch wenn die Seite zu ist oder das Handy schläft. Die Seite
// merkt sich die Job-Kennungen ("gufo-auftraege"); kommt man zurück, fragt sie nach und holt fertige Ergebnisse ab.
// Video-Aufträge stehen wie bisher in "gufo-video-jobs" (pages/video) und werden hier nur mit angezeigt.
(function () {
  const KEY = 'gufo-auftraege', VIDEO_KEY = 'gufo-video-jobs', GERAET_KEY = 'gufo-geraet';
  const lesen = k => { try { return JSON.parse(localStorage.getItem(k)) || []; } catch (e) { return []; } };
  const schreiben = (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} };
  const ENDE = ['completed', 'failed', 'cancelled', 'expired'];

  function geraet() {
    let g = null;
    try { g = localStorage.getItem(GERAET_KEY); } catch (e) {}
    if (!g) {
      g = 'g-' + (crypto.randomUUID ? crypto.randomUUID() : Date.now().toString(36) + Math.random().toString(36).slice(2));
      try { localStorage.setItem(GERAET_KEY, g); } catch (e) {}
    }
    return g;
  }

  // alle gemerkten Aufträge dieses Geräts, neueste zuerst: {id, art: 'job'|'video', seite, url, titel, start}
  function liste() {
    const jobs = lesen(KEY);
    const videos = lesen(VIDEO_KEY).map(j => ({ id: j.id, art: 'video', seite: 'video', url: j.url || '', start: j.start,
      titel: j.info ? `${j.info.len} s · ${j.info.label} · Seed ${j.info.seed}` : j.id }));
    return [...jobs, ...videos].sort((a, b) => (b.start || 0) - (a.start || 0));
  }
  function merken(e) { schreiben(KEY, [...lesen(KEY).filter(x => x.id !== e.id), e]); }
  function vergessen(id) {
    schreiben(KEY, lesen(KEY).filter(x => x.id !== id));
    schreiben(VIDEO_KEY, lesen(VIDEO_KEY).filter(x => x.id !== id));
  }

  // Texte über gemeinsam/sprache.js (de.ini/en.ini, Abschnitt [gemeinsam]); ohne sie bleibt der deutsche Text.
  const T = (k, d, v) => window.gufoT ? gufoT(k, d, v) : d.replace(/\{(\w+)\}/g, (m, n) => (v && n in v ? v[n] : m));

  // Vorrang (Router 2.0.3, 13.11.): 0 = normal, 5 = hoch, 9 = sofort; je Gerät gemerkt (localStorage "gufo-vorrang").
  // Geht als Kopf X-Gufo-Priority und Feld "priority" mit jedem Auftrag (POST /v1/jobs). Der Router rechnet höhere Werte
  // zuerst, bei gleichem Wert den älteren Auftrag – über beide Rechner (jobs.hpp „Priority“); ein älterer Router kennt
  // das Feld nicht und ordnet wie bisher nach Eingang.
  const VORRANG_KEY = 'gufo-vorrang';
  const VORRANG_STUFEN = [[0, 'vorrang.normal', 'normal'], [5, 'vorrang.hoch', 'hoch'], [9, 'vorrang.sofort', 'sofort']];
  const VORRANG_SEITEN = ['bild', 'video', 'musik', 'objekt', 'hochskalieren'];
  function vorrang() {
    let v = 0;
    try { v = parseInt(localStorage.getItem(VORRANG_KEY), 10) || 0; } catch (e) {}
    return VORRANG_STUFEN.some(s => s[0] === v) ? v : 0;
  }
  function vorrangSetzen(wert) {
    const v = VORRANG_STUFEN.some(s => s[0] === Number(wert)) ? Number(wert) : 0;
    try { if (v > 0) localStorage.setItem(VORRANG_KEY, String(v)); else localStorage.removeItem(VORRANG_KEY); } catch (e) {}
    document.querySelectorAll('select.vorrang').forEach(s => { s.value = String(v); });
    return v;
  }
  // Kopfzeile(n) für eine Anfrage an den Router: {'X-Gufo-Priority': '5'} oder {} bei normal. Auch für direkte Anfragen
  // (z. B. /v1/images/generations ohne Auftrag): der Router lässt wartende Pool-Anfragen mit höherem Vorrang zuerst.
  function vorrangKopf(wert) {
    const v = typeof wert === 'number' ? wert : vorrang();
    return v > 0 ? { 'X-Gufo-Priority': String(v) } : {};
  }
  // Bedienelement „Vorrang“: <label class="vorrang"><span>Vorrang</span> <select>normal|hoch|sofort</select></label>.
  // Texte über data-t (sprache.js übersetzt sie beim Laden der Sprache). Liefert das Label; ohne el wird es nur gebaut.
  function vorrangFeld(el) {
    if (!document.getElementById('gufo-vorrang-stil')) {
      const stil = document.createElement('style'); stil.id = 'gufo-vorrang-stil';
      stil.textContent = 'label.vorrang{display:inline-flex;align-items:center;gap:6px;margin-left:8px;font-size:13px;color:var(--muted,#6b6b70)}label.vorrang select{font:inherit;padding:3px 6px}';
      document.head.append(stil);
    }
    const label = document.createElement('label'); label.className = 'vorrang';
    const text = document.createElement('span'); text.dataset.t = 'gemeinsam.vorrang'; text.textContent = T('gemeinsam.vorrang', 'Vorrang');
    const sel = document.createElement('select'); sel.className = 'vorrang';
    sel.dataset.tTitle = 'gemeinsam.vorrang.tipp';
    sel.title = T('gemeinsam.vorrang.tipp', 'Reihenfolge im Pool beider Rechner: sofort (9) vor hoch (5) vor normal; bei gleichem Vorrang der ältere Auftrag zuerst. Gilt für die nächsten Aufträge dieses Geräts.');
    for (const [wert, key, de] of VORRANG_STUFEN) {
      const o = document.createElement('option'); o.value = String(wert); o.dataset.t = 'gemeinsam.' + key; o.textContent = T('gemeinsam.' + key, de);
      sel.append(o);
    }
    sel.value = String(vorrang());
    sel.addEventListener('change', () => vorrangSetzen(sel.value));
    label.append(text, sel);
    if (el) el.append(label);
    return label;
  }
  // Auf den Seiten Bild, Video, Musik, Objekt und Hochskalieren steht das Feld hinter „Abbrechen“ neben dem Knopf #go,
  // ohne Umbau der Seiten; eine Seite, die es selbst einbaut (label.vorrang vorhanden), bekommt kein zweites.
  function vorrangEinbauen() {
    const seite = document.documentElement.dataset.seite, go = document.getElementById('go');
    if (!go || !VORRANG_SEITEN.includes(seite) || document.querySelector('label.vorrang')) return;
    const neben = document.getElementById('cancel') || go;
    neben.insertAdjacentElement('afterend', vorrangFeld(null));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', vorrangEinbauen); else vorrangEinbauen();

  // Fehler mit HTTP-Status (0 = Netz) und Servertext; klartext() macht daraus einen verständlichen Satz (11.11.).
  function fehlerMit(status, text, code) { const err = new Error(text || `HTTP ${status}`); err.status = status; if (code) err.code = code; return err; }
  function klartext(err) {
    const st = err && err.status, msg = (err && err.message) || '';
    if (st === 0) return T('gemeinsam.fehler.netz', 'Router nicht erreichbar.');
    // Router ≥ 1.4.20: kein Router kann den Auftrag (z. B. LoRA nur auf dem anderen Rechner). Der Servertext nennt schon,
    // welcher Router welche LoRA hat; dazu der Tipp, was zu tun ist.
    if (st === 400 && err.code === 'no_capable_router') return msg + ' – ' + T('gemeinsam.fehler.keinrouter', 'LoRA auf dem anderen Rechner eintragen oder ohne LoRA.');
    if (st === 507) return T('gemeinsam.fehler.507', 'Nicht genug Speicher auf diesem Rechner – ein anderes Modell ist noch geladen oder der Rechner ist voll. Nochmal versuchen hilft oft nach ein paar Minuten.');
    if (st === 503) return T('gemeinsam.fehler.503', 'Modell lädt gerade oder der andere Rechner ist beschäftigt – bitte kurz warten.');
    return msg;   // 400, 404 (Pausenmeldung) und alles andere: Servertext zeigen
  }
  // Fehler in ein Status-Element schreiben: das Element wird zum Fehlerkasten (Klassen err + fehlerkasten, Stil aus
  // gemeinsam/sprache.js, 12.11.); mit nochmal() kommt ein Knopf „Nochmal“, der denselben Auftrag neu startet. Sobald die
  // Seite die Klasse err wieder entfernt (normaler Stand), ist der Kasten weg.
  function fehlerZeigen(el, err, nochmal) {
    el.classList.add('err', 'fehlerkasten');
    const text = document.createElement('span');
    text.textContent = T('gemeinsam.fehler', 'Fehler: {fehler}', { fehler: klartext(err) });
    el.textContent = ''; el.append(text);
    if (!nochmal) return;
    const b = document.createElement('button'); b.type = 'button'; b.className = 'ghost'; b.textContent = T('gemeinsam.nochmal', 'Nochmal');
    b.onclick = () => { el.classList.remove('err'); el.textContent = ''; nochmal(); };
    el.append(b);
  }
  // Fortschritt eines Stands als Zusatztext für den Sekundenzähler (B1): " · wartet", " · 42 %" oder "" (kein Fortschritt
  // gemeldet, z. B. Musik/Stimme ohne Schritte). bar (optional, <progress>) wird nur gezeigt, wenn Prozent da sind.
  function fortschritt(s, bar) {
    if (bar) bar.hidden = !(s && !s.fertig && s.progress > 0);
    if (!s || s.fertig) return '';
    const wo = warteschlange(s.view);   // Router ≥ 1.4.20: Position in der gemeinsamen Warteschlange / rechnet auf dem anderen Rechner
    if (s.status === 'queued') return wo ? ' · ' + wo : T('gemeinsam.phase.wartet', ' · wartet (Modell lädt oder anderer Auftrag dran)');
    if (s.progress > 0) { if (bar) bar.value = s.progress; return T('gemeinsam.phase.prozent', ' · {p} %', { p: Math.round(s.progress) }) + (wo ? ' · ' + wo : ''); }
    return wo ? ' · ' + wo : '';
  }
  // Gemeinsame Warteschlange beider Router (Router ≥ 1.4.20, 12.11.): das Job-JSON hat "position" (1 = ältester wartender
  // Auftrag beider Rechner, null = wartet nicht) und "running_on" ("local", Partner-URL oder null). Liefert den passenden
  // Satz oder '' – ein älterer Router (1.4.19) hat die Felder nicht, dann bleibt alles wie bisher.
  // kurzName: ".56" aus der Partner-URL (letztes Oktett); ohne IPv4 der Hostname.
  function kurzName(url) {
    const s = String(url || ''), m = /(\d+)\.(\d+)\.(\d+)\.(\d+)/.exec(s);
    if (m) return '.' + m[4];
    try { return new URL(s).hostname || s; } catch (e) { return s; }
  }
  function warteschlange(d) {
    if (!d || typeof d !== 'object') return '';
    if (typeof d.running_on === 'string' && d.running_on && d.running_on !== 'local')
      return T('gemeinsam.pool.dort', 'rechnet auf dem zweiten Rechner ({name})', { name: kurzName(d.running_on) });
    // Router 2.0.3: "priority" > 0 wird mit angezeigt (Vorrang), damit man sieht, warum ein Auftrag vorgezogen wird.
    const prio = Number(d.priority) > 0 ? T('gemeinsam.pool.vorrang', ' · Vorrang {p}', { p: d.priority }) : '';
    if (d.position != null && Number(d.position) > 0 && d.status !== 'in_progress')
      return T('gemeinsam.pool.position', 'Position {p} in der gemeinsamen Warteschlange', { p: d.position }) + prio;
    return '';
  }

  async function base64(buf) {
    const bytes = new Uint8Array(buf); let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }

  // Auftrag als Job abschicken. json ODER formData (multipart, z. B. Bildbearbeitung). Antwort: der gemerkte Eintrag.
  // angaben (optional): Prompt, Seed, Modell und Seitenfelder für die Ablage; landen in job.metadata (Router: Objekt bis 16 KiB).
  // vorrang (optional, 0–9): sonst die Auswahl des Geräts (Feld „Vorrang“), aber nur auf den Seiten, die das Feld zeigen
  // (VORRANG_SEITEN) – Stimme, Lipsync usw. schicken normal. Geht als Kopf X-Gufo-Priority und Feld "priority".
  async function absenden({ url, path, json, formData, seite, titel, extra, angaben, vorrang: vorrangWert }) {
    const request = { method: 'POST', path };
    if (formData) {
      const r = new Response(formData);
      request.content_type = r.headers.get('content-type');
      request.body_base64 = await base64(await r.arrayBuffer());
    } else request.body = json;
    const prio = typeof vorrangWert === 'number' ? vorrangWert : (VORRANG_SEITEN.includes(seite) ? vorrang() : 0);
    let res;
    try {
      res = await fetch(url + '/v1/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json', ...vorrangKopf(prio) },
        body: JSON.stringify({ request, ...(prio > 0 ? { priority: prio } : {}), metadata: { geraet: geraet(), seite, titel, ...(angaben || {}) } }) });
    } catch (err) { throw fehlerMit(0, err.message); }   // Netzfehler: Router aus oder nicht erreichbar
    const d = await res.json().catch(() => ({}));
    // Pausen-Router (pause-run.ps1) hat kein /v1/jobs -> 404: verständlich melden statt leerem Pfad (05.11.).
    if (res.status === 404) throw fehlerMit(404, T('gemeinsam.fehler.404', 'Dieser Router nimmt gerade keine Aufträge an (Wartungspause oder ohne Auftragsliste --jobs-dir). Bitte später erneut versuchen oder den anderen Rechner wählen.'));
    if (!res.ok) throw fehlerMit(res.status, d?.error?.message || `HTTP ${res.status}`, d?.error?.code);
    const e = { id: d.id, art: 'job', seite, url, titel, start: Date.now(), ...(prio > 0 ? { vorrang: prio } : {}), ...(extra ? { extra } : {}) };
    merken(e);
    return e;
  }

  // Stand eines Eintrags: {status, progress, fertig, fehler, view}; null = Router gerade nicht erreichbar
  async function stand(e) {
    let r;
    try { r = await fetch((e.url || '') + (e.art === 'video' ? '/v1/videos/' : '/v1/jobs/') + e.id, { cache: 'no-store' }); }
    catch (err) { return null; }
    if (r.status === 404) return { status: 'weg', progress: 0, fertig: true, fehler: 'nicht mehr auf dem Router', view: null };
    const d = await r.json().catch(() => ({}));
    if (!r.ok) return null;
    const status = d.status;
    const fehler = ['failed', 'cancelled', 'expired'].includes(status) ? (d.error?.message || status) : '';
    return { status, progress: d.progress || 0, fertig: ENDE.includes(status), fehler, view: d };
  }

  // so lange nachfragen, bis der Auftrag fertig ist; beiStand(s) bei jeder Antwort; liefert den letzten Stand
  async function folgen(e, beiStand, ms = 2000) {
    for (;;) {
      const s = await stand(e);
      if (beiStand) beiStand(s);
      if (s && s.fertig) return s;
      await new Promise(r => setTimeout(r, s ? ms : 5000));
    }
  }

  // Job abwarten und die erste Ergebnisdatei holen, deren Name zum Muster passt (z. B. /^output\./ für WAV,
  // /^image-/ für Bilder). Bei Fehler wird der Eintrag vergessen und ein Error geworfen.
  async function ergebnis(e, muster, beiStand) {
    const s = await folgen(e, beiStand);
    if (s.fehler) { vergessen(e.id); throw fehlerMit(s.view?.error?.status_code || -1, s.fehler); }
    const datei = (s.view?.result?.files || []).find(f => muster.test(f.name));
    if (!datei) { vergessen(e.id); throw new Error(s.view?.result?.response?.error?.message || 'kein Ergebnis in der Antwort'); }
    const r = await fetch((e.url || '') + datei.url);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.blob();
  }

  // Auftrag abbrechen (12.11., UX-Bericht Nr. 6): POST /v1/jobs/{id}/cancel. true = Router hat angenommen. Die laufende
  // folgen()-Schleife sieht danach den Stand "cancelled" und endet mit fehler – die Seite zeigt dann „abgebrochen“.
  async function abbrechen(e) {
    if (!e || !e.id || e.art === 'video') return false;
    try {
      const r = await fetch((e.url || '') + '/v1/jobs/' + encodeURIComponent(e.id) + '/cancel', { method: 'POST' });
      return r.ok;
    } catch (err) { return false; }
  }

  // Aufträge einer Seite, die noch nicht abgeholt sind (Seite war zu oder Gerät hat geschlafen)
  function offen(seite) { return liste().filter(e => e.seite === seite && e.art === 'job' && e.extra); }

  window.gufoAuftraege = { geraet, liste, merken, vergessen, absenden, stand, folgen, ergebnis, offen, klartext, fehlerZeigen, fortschritt, warteschlange, kurzName, abbrechen,
    vorrang, vorrangSetzen, vorrangKopf, vorrangFeld };
})();
