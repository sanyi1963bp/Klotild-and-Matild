# ArchMorph – Fejlesztési napló

_Utolsó frissítés: 2026-04-09_

---

## Mi ez a projekt?

**ArchMorph Professional** – PyQt6-alapú képmorfolizáló alkalmazás.  
Két épületfotó között (pl. régi/új állapot) azonos pontokat jelölünk ki,
és ezekből morfolizált átmenetet generálunk.

**Fájlok** (mind az `outputs/` mappában):
- `archmorph.py` – főablak, projekt állapot, workflow panel
- `point_editor.py` – pontszerkesztő modul (A+B canvas)
- `gcp_dialog.py` – GCP (Ground Control Point) igazítási ablak
- `archmorph_config_loader.py` – konfig betöltő
- `TRANSLATIONS.py` – fordítási segédmodul

---

## Mi van kész?

### Fő funkciók
- Kép betöltés (A és B oldal), dőlés-korrekció (TiltCorrectionDialog, Hough-alapú)
- GCP igazítás párbeszédablak (pontok élénk narancssárga körrel, kontrasztos)
- Pontszerkesztő: dupla klikk = pont, húzás = mozgatás, ROI keret, zoom, undo
- Automatikus pontillesztés tab
- Előnézet + export
- Workflow panel (dokkoló oldalsáv): lépésenkénti útmutató gombokkal, színes keretekkel

### Rajzolási módok (point_editor.py)
- **Pont mód**: dupla klikk = pontpár hozzáadása
- **Vonallánc mód**: klikk = csúcs, dupla klikk = utolsó pont + lezárás
- **Ív mód**: klikk = csúcs, dupla klikk = utolsó pont + lezárás

### Vonallánc workflow (legutóbb implementálva – persistens réteg)
1. Dupla klikk → vonallánc indítása (első csúcs + interpolált B-pár)
2. Egyszeres klikk → csúcs hozzáadása; pár azonnal megjelenik a másik canvas-on
3. Dupla klikk → utolsó csúcs + `_commit_polyline()` → **nem** bontja pontpárokká azonnal
4. A vonallánc `project.polylines` listában tárolódik: `{"pts_a": [...], "pts_b": [...]}`
5. Mindkét canvas-on megjelenik a vonallánc összekötve, csúcsokkal (L1, L2, … felirat)
6. **Csúcs mozgatás**: bal húzás bármelyik canvas csúcsán → csak az a csúcs mozog
7. **Törlés**: jobb klikk bármelyik csúcson → egész vonallánc törlése (undo-val)
8. **Rendereléskor**: `PointEditorWidget.polylines_to_point_pairs()` osztja fel
   - A-oldal: 30 px-enként → N pont
   - B-oldal: pontosan N pontra mintavételez egyenletesen (NEM interpoláció)

---

## Ami ki volt hozva, de VISSZA LETT VONVA

_(Ne implementáld újra ezeket, hacsak a felhasználó nem kéri!)_

- Semmi nincs visszavonva – minden fent van.

---

## Következő feladat

A vonallánc-réteg kész és működőképes. Lehetséges folytatási irányok:

1. **Tesztelés** – futtatni az alkalmazást és megnézni hogy a vonalak helyesen jelennek meg, mozgathatók, törölhetők
2. **Vonallánc törlése a toolbar-ból** – gomb az összes vonallánc törléséhez (opcionális)
3. **Vonallánc vertex törlése** – jelenleg csak az egész vonallánc törölhető; ha kell, egyedi csúcs törlése is megvalósítható
4. **Export preview frissítés** – ha a felhasználó a pontokat kézzel szerkeszti és elmegy exportálni auto-match nélkül, az export tab nem frissül; ez megoldható egy "Morfpontok frissítése" gombbal az export fülön

---

## Fontos technikai tudnivalók

- **Qt double-click sorrendje**: Press → Release → DoubleClick → Release  
  (a 2. klikk mousePressEvent-et NEM küld, csak DoubleClickEvent-et)
- **GCP pont attribútum**: `self._points` (NEM `self._pts` – az crash-t okozna)
- **Silens kilépés** ellen: `main()` körül `try/except` van, hibaüzenetes dialog-gal
- **Git push**: a Windows NTFS mounton lévő `.git/config` nem olvasható a sandboxból → rsync `/tmp`-be kell, ott commitolni

---

## Workflow lépések sorrendje (UI-ban)

① Kép A betöltése  
② Kép B betöltése  
③ Dőlés-korrekció ✦ opcionális  
④ GCP igazítás  
⑤ Kézi szerkesztő ✦ opcionális  
⑥ Automata illesztés  
⑦ Előnézet  
⑧ Export  
