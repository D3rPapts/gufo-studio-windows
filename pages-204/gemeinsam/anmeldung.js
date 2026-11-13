// gufo-Seiten: Anmeldung (2.0.3, Nutzerverwaltung). Ein Baustein für alle Seiten, nach sprache.js einbinden:
//   <script src="../gemeinsam/anmeldung.js"></script>
// Holt GET /router/me. Antwort 200: window.gufoNutzer = {name, rolle, bereiche, filter, auth, quelle, verwaltung};
// zeigt rechts in der Leiste Name + „Abmelden“ (nur bei Anmeldung per Cookie/Token; bei auth = off oder lokal bleibt
// die Leiste wie bisher), blendet Kacheln (Startseite, a.kachel[data-pfad]) und Menüpunkte aus, deren Bereich der
// Nutzer nicht darf, und zeigt Admins den Menüpunkt „Verwaltung“. Antwort 401: weiter zu /login/?zurueck=<diese Seite>.
// Außerdem: antwortet der eigene Router später mit 401 (Sitzung abgelaufen), geht es ebenfalls zur Anmeldung.
// Texte: [gemeinsam] anmeldung.* (sprache.js teilt nur [gemeinsam] und [faehigkeiten] mit allen Seiten).
// Ereignis window "gufo-nutzer" (detail = Nutzer), wenn /router/me gelesen ist. window.gufoDarf(pfad) = darf der
// Nutzer die Seite <pfad> (Ordnername)?
// 2.0.4: „Passwort ändern“ neben „Abmelden“ (nur Rolle admin/nutzer mit Cookie-Sitzung, nie lokal/Partner):
// kleiner Dialog (altes, neues, Wiederholung) → POST /router/me/passwort {alt, neu}; Texte [gemeinsam] anmeldung.passwort.*.
(function () {
  const T = (k, d, v) => window.gufoT ? gufoT('gemeinsam.anmeldung.' + k, d, v) : d;
  // Seite (Ordnername, deutsch und englisch) -> Bereich wie im Router (auth.cpp PageArea).
  const BEREICH = { bild: 'bild', picture: 'bild', hochskalieren: 'bild', schritte: 'bild', steps: 'bild',
                    video: 'video', lipsync: 'video', objekt: '3d', object: '3d',
                    stimme: 'stimme', voice: 'stimme', stimmwandel: 'stimme', transkript: 'stimme', transcript: 'stimme', hoertest: 'stimme',
                    musik: 'musik', music: 'musik', ablage: 'ablage', ergebnisse: 'ablage', chat: 'text',
                    speicher: 'status', memory: 'status', pool: 'status', status: 'status', nacht: 'status', night: 'status' };
  const hier = () => location.pathname + location.search;
  const zurLogin = () => {
    if (/^\/login\/?/.test(location.pathname)) return;
    location.href = '/login/?zurueck=' + encodeURIComponent(hier());
  };
  let nutzer = null;

  function darf(pfad) {
    if (!nutzer || nutzer.auth === 'off' || nutzer.rolle === 'admin') return true;
    const b = BEREICH[pfad];
    return !b || (nutzer.bereiche || []).includes(b);
  }

  // Kacheln (Startseite) und Menüpunkte, die der Nutzer nicht darf, verschwinden; der Rest bleibt wie er ist.
  function ausblenden() {
    if (!nutzer || nutzer.auth === 'off') return;
    for (const a of document.querySelectorAll('a.kachel[data-pfad], .gufo-navi-links a[data-pfad]')) {
      if (!darf(a.dataset.pfad)) a.hidden = true;
    }
    // Gruppen ohne sichtbare Kachel (Startseite): Überschrift mit ausblenden.
    for (const gruppe of document.querySelectorAll('.kacheln')) {
      if (![...gruppe.querySelectorAll('a.kachel')].some(a => !a.hidden)) {
        gruppe.hidden = true;
        const h = gruppe.previousElementSibling;
        if (h && /^H[1-6]$/.test(h.tagName)) h.hidden = true;
      }
    }
  }

  function leiste() {
    if (!nutzer || document.querySelector('.gufo-nutzer')) return;
    const quelle = nutzer.quelle;
    const bar = document.querySelector('.gufo-navi');
    if (!bar) return;
    // Verwaltung: Menüpunkt für Admins (Sitzung oder lokal), auch bei auth = off vom Rechner selbst.
    if (nutzer.verwaltung || quelle === 'loopback' || (quelle === 'off' && /^(127\.|localhost$|\[::1\]$)/.test(location.hostname))) {
      const links = bar.querySelector('.gufo-navi-links');
      if (links && !links.querySelector('a[data-pfad="verwaltung"]')) {
        const a = document.createElement('a'); a.dataset.pfad = 'verwaltung'; a.href = '/verwaltung/';
        a.dataset.t = 'gemeinsam.navi.verwaltung'; a.textContent = T('verwaltung', 'Verwaltung');
        if (/^\/verwaltung\//.test(location.pathname)) a.setAttribute('aria-current', 'page');
        links.append(a);
      }
    }
    if (quelle !== 'cookie' && quelle !== 'bearer') return;   // off/lokal: Leiste wie bisher
    const box = document.createElement('span'); box.className = 'gufo-nutzer';
    const name = document.createElement('span'); name.className = 'gufo-nutzer-name';
    name.textContent = nutzer.name; name.title = T('rolle.' + nutzer.rolle, nutzer.rolle);
    const ab = document.createElement('button'); ab.type = 'button'; ab.className = 'gufo-nutzer-ab';
    ab.textContent = T('abmelden', 'Abmelden'); ab.dataset.t = 'gemeinsam.anmeldung.abmelden';
    ab.onclick = async () => {
      ab.disabled = true;
      try { await fetch('/logout', { method: 'POST', cache: 'no-store' }); } catch (e) {}
      location.href = '/login/';
    };
    box.append(name);
    if (quelle === 'cookie' && (nutzer.rolle === 'admin' || nutzer.rolle === 'nutzer')) {
      const pw = document.createElement('button'); pw.type = 'button'; pw.className = 'gufo-nutzer-ab';
      pw.textContent = T('passwort.knopf', 'Passwort ändern'); pw.dataset.t = 'gemeinsam.anmeldung.passwort.knopf';
      pw.onclick = passwortDialog;
      box.append(pw);
    }
    box.append(ab);
    const sprache = bar.querySelector('.sprachwahl');
    if (sprache) sprache.before(box); else bar.append(box);
    const css = document.createElement('style');
    css.textContent = `.gufo-nutzer{display:inline-flex;align-items:center;gap:6px;margin-left:auto;flex:none;font-size:13px;color:var(--muted,#6b6b70)}
.gufo-nutzer+.sprachwahl{margin-left:0}
.gufo-nutzer-name{font-weight:600;color:var(--text,#1d1d1f);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gufo-nutzer-ab{font:inherit;font-size:13px;min-height:36px;padding:4px 10px;border-radius:8px;cursor:pointer;background:transparent;color:var(--muted,#6b6b70);border:1px solid var(--line,#dcdcd7)}
.gufo-nutzer-ab:hover{color:var(--text,#1d1d1f);border-color:var(--muted,#6b6b70)}
@media (max-width:640px),(pointer:coarse){.gufo-nutzer-ab{min-height:44px}.gufo-nutzer-name{max-width:80px}}`;
    document.head.append(css);
  }

  // Passwort ändern: Dialog mit drei Feldern; Fehler in Klartext, nach Erfolg „gespeichert“.
  function passwortDialog() {
    let dlg = document.querySelector('.gufo-passwort');
    if (!dlg) {
      dlg = document.createElement('dialog'); dlg.className = 'gufo-passwort';
      dlg.innerHTML = '<form method="dialog" class="gufo-passwort-form">'
        + '<h3 data-t="gemeinsam.anmeldung.passwort.titel">Passwort ändern</h3>'
        + '<label><span data-t="gemeinsam.anmeldung.passwort.alt">Altes Passwort</span><input type="password" name="alt" autocomplete="current-password" required></label>'
        + '<label><span data-t="gemeinsam.anmeldung.passwort.neu">Neues Passwort (mind. 4 Zeichen)</span><input type="password" name="neu" autocomplete="new-password" required minlength="4"></label>'
        + '<label><span data-t="gemeinsam.anmeldung.passwort.wieder">Neues Passwort wiederholen</span><input type="password" name="wieder" autocomplete="new-password" required></label>'
        + '<p class="gufo-passwort-meldung" aria-live="polite"></p>'
        + '<div class="gufo-passwort-knoepfe"><button type="submit" class="haupt" data-t="gemeinsam.anmeldung.passwort.speichern">Speichern</button>'
        + '<button type="button" class="zu" data-t="gemeinsam.anmeldung.passwort.abbrechen">Abbrechen</button></div></form>';
      document.body.append(dlg);
      const css = document.createElement('style');
      css.textContent = `.gufo-passwort{border:1px solid var(--line,#dcdcd7);border-radius:12px;background:var(--card,#fff);color:var(--text,#1d1d1f);padding:16px;max-width:min(420px,92vw)}
.gufo-passwort::backdrop{background:rgba(0,0,0,.4)}
.gufo-passwort h3{margin:0 0 8px;font-size:16px}
.gufo-passwort label{display:block;font-size:13px;color:var(--muted,#6b6b70);margin:8px 0 2px}
.gufo-passwort input{display:block;width:100%;box-sizing:border-box;font:inherit;font-size:16px;padding:8px 10px;margin-top:2px;border:1px solid var(--line,#dcdcd7);border-radius:8px;background:var(--bg,#f6f6f4);color:var(--text,#1d1d1f);min-height:44px}
.gufo-passwort-meldung{font-size:14px;min-height:20px;margin:10px 0 0}
.gufo-passwort-meldung.err{color:var(--err,#b3261e);font-weight:600}.gufo-passwort-meldung.ok{color:var(--ok,#1e7a3c);font-weight:600}
.gufo-passwort-knoepfe{display:flex;gap:8px;margin-top:10px}
.gufo-passwort-knoepfe button{font:inherit;font-size:14px;min-height:40px;padding:4px 14px;border-radius:8px;cursor:pointer;background:transparent;color:var(--text,#1d1d1f);border:1px solid var(--line,#dcdcd7)}
.gufo-passwort-knoepfe button.haupt{background:var(--accent,#5b4fc4);color:var(--on-accent,#fff);border-color:var(--accent,#5b4fc4);font-weight:600}
.gufo-passwort-knoepfe button:disabled{opacity:.6;cursor:default}`;
      document.head.append(css);
      const form = dlg.querySelector('form');
      const meldung = dlg.querySelector('.gufo-passwort-meldung');
      const melden = (text, art) => { meldung.textContent = text; meldung.className = 'gufo-passwort-meldung' + (art ? ' ' + art : ''); };
      dlg.querySelector('button.zu').onclick = () => dlg.close();
      form.addEventListener('submit', async ev => {
        ev.preventDefault();
        const alt = form.elements.alt.value, neu = form.elements.neu.value, wieder = form.elements.wieder.value;
        if (neu.length < 4) { melden(T('passwort.kurz', 'Das neue Passwort braucht mindestens 4 Zeichen.'), 'err'); return; }
        if (neu !== wieder) { melden(T('passwort.ungleich', 'Die Wiederholung stimmt nicht mit dem neuen Passwort überein.'), 'err'); return; }
        const knopf = form.querySelector('button.haupt'); knopf.disabled = true;
        melden(T('passwort.laeuft', 'wird gespeichert …'));
        try {
          const r = await fetch('/router/me/passwort', { method: 'POST', cache: 'no-store', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ alt, neu }) });
          const d = await r.json().catch(() => ({}));
          const code = d.error && d.error.code;
          if (r.ok) { melden(T('passwort.ok', 'Gespeichert. Andere Geräte müssen sich neu anmelden.'), 'ok'); form.reset(); setTimeout(() => dlg.close(), 1500); }
          else if (code === 'wrong_password') melden(T('passwort.falsch', 'Das alte Passwort ist falsch.'), 'err');
          else if (code === 'invalid_password') melden(T('passwort.kurz', 'Das neue Passwort braucht mindestens 4 Zeichen.'), 'err');
          else if (r.status === 429) melden(T('passwort.gesperrt', 'Zu viele Fehlversuche, bitte {s} Sekunden warten.', { s: r.headers.get('Retry-After') || 60 }), 'err');
          else melden(T('passwort.fehler', 'Fehler {status}: {text}', { status: r.status, text: (d.error && d.error.message) || '' }), 'err');
        } catch (e) { melden(T('passwort.netz', 'Router nicht erreichbar.'), 'err'); }
        finally { knopf.disabled = false; }
      });
      if (window.gufoSprache) gufoSprache.apply(dlg);
    }
    dlg.querySelector('.gufo-passwort-meldung').textContent = '';
    dlg.querySelector('form').reset();
    dlg.showModal();
    dlg.querySelector('input[name=alt]').focus();
  }

  // Sitzung läuft während der Arbeit ab: 401 vom eigenen Router -> Anmeldeseite (nicht für den Partner-Router).
  const fetch0 = window.fetch;
  window.fetch = async function (eingabe, init) {
    const r = await fetch0.call(this, eingabe, init);
    try {
      if (r.status === 401 && nutzer && nutzer.auth !== 'off') {
        const u = new URL(typeof eingabe === 'string' ? eingabe : eingabe.url, location.href);
        if (u.origin === location.origin && !/^\/(login|logout|router\/me)$/.test(u.pathname)) zurLogin();
      }
    } catch (e) {}
    return r;
  };

  async function start() {
    let r;
    try { r = await fetch0('/router/me', { cache: 'no-store' }); } catch (e) { return; }   // Router weg: Seite zeigt es selbst
    if (r.status === 401) { zurLogin(); return; }
    if (!r.ok) return;                                                               // älterer Router ohne /router/me
    try { nutzer = await r.json(); } catch (e) { return; }
    window.gufoNutzer = nutzer;
    const zeichnen = () => { leiste(); ausblenden(); };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', zeichnen); else zeichnen();
    window.dispatchEvent(new CustomEvent('gufo-nutzer', { detail: nutzer }));
  }
  window.gufoDarf = darf;
  start();
})();
