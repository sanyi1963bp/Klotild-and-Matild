"""
ArchMorph Professional - Hungarian to English Translation Dictionary
Complete UI string translations for all three modules:
- archmorph.py (main application)
- point_editor.py (point editor)
- config_editor.py (configuration editor)
"""

_EN = {
    # ══════════════════════════════════════════════════════════════════════════
    # TAB TITLES AND MAIN MENU ITEMS
    # ══════════════════════════════════════════════════════════════════════════
    "⚡  Automata illesztés": "⚡  Auto Matching",
    "✏️  Pontszerkesztő": "✏️  Point Editor",
    "🔍  Előnézet": "🔍  Preview",
    "🎬  Export": "🎬  Export",
    "⚙  Beállítások": "⚙  Settings",

    # File Menu
    "Fájl": "File",
    "Kép A betöltése…": "Load Image A…",
    "Kép B betöltése…": "Load Image B…",
    "Projekt mentése…": "Save Project…",
    "Projekt megnyitása…": "Open Project…",
    "Kilépés": "Exit",

    # Edit Menu
    "Szerkesztés": "Edit",
    "Visszavonás": "Undo",
    "Összes pontpár törlése": "Delete All Point Pairs",

    # Toolbar Actions
    "📂  Kép A": "📂  Image A",
    "📂  Kép B": "📂  Image B",
    "📂  Mindkét kép…": "📂  Both Images…",
    "💾  Mentés": "💾  Save",
    "📁  Projekt": "📁  Project",
    "Gyorseszközök": "Quick Access Toolbar",

    # Tooltips for Toolbar
    "Kép A betöltése (Ctrl+1)": "Load Image A (Ctrl+1)",
    "Kép B betöltése (Ctrl+2)": "Load Image B (Ctrl+2)",
    "Projekt mentése (Ctrl+S)": "Save Project (Ctrl+S)",
    "Projekt megnyitása (Ctrl+O)": "Open Project (Ctrl+O)",

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOMATIC MATCHING TAB
    # ══════════════════════════════════════════════════════════════════════════
    "Illesztési algoritmus (backend)": "Matching Algorithm (Backend)",
    "Backend paraméterek": "Backend Parameters",
    "Közös beállítások": "Common Settings",

    # Backend Selection
    "Backend:": "Backend:",
    "SuperPoint + LightGlue": "SuperPoint + LightGlue",
    "DISK + LightGlue": "DISK + LightGlue",
    "LoFTR (kornia)": "LoFTR (kornia)",
    "SIFT (OpenCV)": "SIFT (OpenCV)",

    # SuperPoint Parameters
    "Max. kulcspontok:": "Max. Keypoints:",
    "Detekciós küszöb:": "Detection Threshold:",
    "NMS sugár (px):": "NMS Radius (px):",
    "Match küszöb (LG):": "Match Threshold (LG):",
    "Mélység-konfidencia:": "Depth Confidence:",
    "Szélesség-konfidencia:": "Width Confidence:",

    # DISK Parameters
    "Match küszöb (LG)": "Match Threshold (LG)",

    # LoFTR Parameters
    "Előtanított modell:": "Pretrained Model:",
    "Konfidencia küszöb:": "Confidence Threshold:",

    # SIFT Parameters
    "Oktáv-rétegek:": "Octave Layers:",
    "Kontraszt küszöb:": "Contrast Threshold:",
    "Gauss sigma:": "Gauss Sigma:",
    "Él küszöb:": "Edge Threshold:",
    "Lowe arány küszöb:": "Lowe Ratio Threshold:",

    # RANSAC Filtering
    "RANSAC szűrés bekapcsolva": "RANSAC Filtering Enabled",
    "Bekapcsolva": "Enabled",
    "Reproj. küszöb:": "Reproj. Threshold:",
    "Reproj. küszöb (px):": "Reproj. Threshold (px):",
    "CPU kényszerítése (debug / GPU nélkül)": "Force CPU (debug / no GPU)",

    # ══════════════════════════════════════════════════════════════════════════
    # POINT EDITOR TAB
    # ══════════════════════════════════════════════════════════════════════════
    "Pontpárok:": "Point Pairs:",
    "Pontpárok: 0": "Point Pairs: 0",
    "Pontpárok: {n}": "Point Pairs: {n}",
    "Kép  A": "Image  A",
    "Kép  B": "Image  B",

    # Point Editor Tooltips
    "Mindkét canvas ROI-jának törlése": "Clear ROI for both canvases",
    "Ctrl+húzás: ROI keret  |  ROI-n: húzás=mozgat, sarok=átméretez  |  Ctrl+klikk: sokszög-ROI  |  Delete: törlés  |  Görgetés: zoom": "Ctrl+drag: ROI frame  |  On ROI: drag=move, corner=resize  |  Ctrl+click: polygon-ROI  |  Delete: remove  |  Scroll: zoom",

    # Context Menu Actions
    "Kiválasztás": "Select",
    "Törlés  (párjával együtt)": "Delete  (with pair)",
    "Meglévő pontok törlése a területen belül": "Delete existing points in area",

    # ══════════════════════════════════════════════════════════════════════════
    # PREVIEW TAB
    # ══════════════════════════════════════════════════════════════════════════
    "Nézet:": "View:",
    "Overlay": "Overlay",
    "Célkép": "Target Image",
    "Warpolt A": "Warped A",
    "Átlátszóság:": "Transparency:",

    # ══════════════════════════════════════════════════════════════════════════
    # EXPORT TAB
    # ══════════════════════════════════════════════════════════════════════════
    "Képkockák": "Frames",
    "Darab:": "Count:",
    "Hány közbülső képkocka legyen az animációban": "Number of intermediate frames in animation",

    "Morph módszer": "Morph Method",
    "Delaunay háromszög": "Delaunay Triangulation",
    "Optikai folyam": "Optical Flow",
    "Homográfia": "Homography",

    "Pontpárok:": "Point Pairs:",
    "{n_pts} pontpár  (min. 3 szükséges)": "{n_pts} point pairs  (min. 3 required)",
    "{n_pts} pontpár elérhető  ✓": "{n_pts} point pairs available  ✓",

    # Flow Quality Presets
    "Minőség:": "Quality:",
    "Gyors": "Fast",
    "Normál": "Normal",
    "Részletes": "Detailed",

    # Easing Curves
    "Animáció": "Animation",
    "Easing görbe:": "Easing Curve:",
    "Lineáris": "Linear",
    "Lassú start": "Ease In",
    "Lassú vége": "Ease Out",
    "S-görbe": "S-Curve",

    # Generation and Export
    "Generálás": "Generation",
    "⚙  Képkockák generálása": "⚙  Generate Frames",
    "⚙  Újragenerálás szükséges!": "⚙  Regeneration Required!",
    "⚠  A beállítások megváltoztak – az előnézet elavult.": "⚠  Settings changed – preview is stale.",

    "▶  Lejátszás": "▶  Play",
    "Animáció mentése MP4 videóként (OpenCV)": "Save animation as MP4 video (OpenCV)",
    "Animáció mentése animált GIF-ként (Pillow)": "Save animation as animated GIF (Pillow)",
    "Minden képkocka mentése külön PNG fájlként": "Save each frame as separate PNG file",

    "MP4 mentése": "Save MP4",
    "GIF mentése": "Save GIF",
    "Mappa kijelölése a PNG sorozatnak": "Select folder for PNG sequence",

    "Animált GIF (*.gif);;Minden fájl (*)": "Animated GIF (*.gif);;All Files (*)",
    "MP4 videó (*.mp4);;Minden fájl (*)": "MP4 Video (*.mp4);;All Files (*)",
    "Képfájlok (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Minden fájl (*)": "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",

    # ══════════════════════════════════════════════════════════════════════════
    # SETTINGS / CONFIG EDITOR TAB
    # ══════════════════════════════════════════════════════════════════════════

    # Color Theme - Points
    "Normál": "Normal",
    "Aktív": "Active",
    "Multi-kijelölt": "Multi-Selected",
    "Sorszám szín": "Index Color",
    "Sorszám (aktív)": "Index (Active)",

    # Color Theme - Points Fill
    "Normál fill": "Normal Fill",
    "Aktív fill": "Active Fill",
    "Multi fill": "Multi Fill",

    # Color Theme - ROI
    "Keret szín": "Border Color",
    "Fogópont": "Handle",
    "Belső fill": "Inner Fill",
    "Külső árnyék": "Outer Shadow",

    # Color Theme - Canvas
    "Háttér": "Background",
    "Keret": "Border",
    "Helytartó": "Placeholder",
    "Kijelölés keret": "Selection Frame",
    "Rajzolás": "Drawing",
    "Lezárt": "Closed",
    "Kitöltés": "Fill",

    # Color Theme - UI Global
    "Ablak háttér": "Window Background",
    "Panel háttér": "Panel Background",
    "Gomb": "Button",
    "Gomb (hover)": "Button (Hover)",
    "Beviteli mező": "Input Field",
    "Elválasztó": "Separator",
    "Fő szöveg": "Main Text",
    "Halvány szöveg": "Dim Text",
    "Felirat": "Label",
    "Csoport fejléc": "Group Header",
    "Elavult gomb h.": "Stale Button Bg.",
    "Elavult gomb k.": "Stale Button Border",
    "Elavult gomb sz.": "Stale Button Text",

    # Sizes and Hit Radii
    "Pont sugár – normál": "Point Radius – Normal",
    "Pont sugár – aktív": "Point Radius – Active",
    "Pont sugár – multi": "Point Radius – Multi",
    "Kattintási érzékenység": "Click Sensitivity",
    "ROI fogópont sugara": "ROI Handle Radius",
    "Minimum ROI méret": "Minimum ROI Size",

    # SuperPoint Config
    "Max. kulcspontok": "Max. Keypoints",
    "Detekciós küszöb": "Detection Threshold",
    "NMS sugár (px)": "NMS Radius (px)",
    "Match küszöb (LG)": "Match Threshold (LG)",
    "Mélység konfidencia": "Depth Confidence",
    "Szélesség konfidencia": "Width Confidence",

    # DISK Config
    # (reuses some from above)

    # LoFTR Config
    "Modell típus": "Model Type",
    "Konfidencia küszöb": "Confidence Threshold",

    # SIFT Config
    "Oktáv-rétegek": "Octave Layers",
    "Kontraszt küszöb": "Contrast Threshold",
    "Gauss sigma": "Gauss Sigma",
    "Lowe arány küszöb": "Lowe Ratio Threshold",

    # RANSAC Config
    "Bekapcsolva": "Enabled",
    "Reproj. küszöb": "Reproj. Threshold",

    # Export Defaults
    "Képkockák száma": "Frame Count",
    "Morph módszer": "Morph Method",
    "Easing görbe": "Easing Curve",
    "Ping-pong": "Ping-Pong",

    # Optical Flow Presets
    "Piramis arány": "Pyramid Scale",
    "Piramisszintek": "Pyramid Levels",
    "Ablakméret (px)": "Window Size (px)",
    "Iterációk": "Iterations",
    "Poly sigma": "Poly Sigma",
    "Poly N": "Poly N",

    # Config Editor Actions
    "💾 Mentés": "💾 Save",
    "📂 Betöltés": "📂 Load",
    "🗑 Törlés": "🗑 Delete",
    "Szín kiválasztása": "Choose Color",

    "Aktuális beállítások mentése profilként": "Save current settings as profile",
    "Kiválasztott profil betöltése": "Load selected profile",
    "Kiválasztott profil törlése": "Delete selected profile",
    "Minden értéket visszaállít az alapértelmezetthez": "Reset all values to defaults",

    "Jobb klikk: visszaállítás alapértelmezettre": "Right-click: reset to default",

    # ══════════════════════════════════════════════════════════════════════════
    # DIALOG TITLES AND MESSAGES
    # ══════════════════════════════════════════════════════════════════════════

    # File Dialogs
    "Kép A megnyitása": "Open Image A",
    "Kép B megnyitása": "Open Image B",
    "Projekt mentése": "Save Project",
    "Projekt megnyitása": "Open Project",
    "Két kép kijelölése (első = A, második = B)": "Select two images (first = A, second = B)",

    # Status Bar Messages
    "Kész.": "Ready.",
    "Automatikus illesztés folyamatban  [{backend}]…": "Auto matching in progress  [{backend}]…",
    "Illesztés kész  –  {n} pontpár  |  eszköz: {device}": "Matching complete  –  {n} point pairs  |  device: {device}",
    "Illesztés megszakítva.": "Matching cancelled.",
    "Illesztés sikertelen.": "Matching failed.",

    "ROI illesztés folyamatban  [{backend}]  ": "ROI matching in progress  [{backend}]  ",
    "ROI illesztés kész  –  +{n_new} új pár  ": "ROI matching complete  –  +{n_new} new pairs  ",
    "ROI illesztés: nem talált egyezést.": "ROI matching: no matches found.",
    "ROI illesztés: a területen belül nincs egyezés.": "ROI matching: no matches in area.",
    "ROI illesztés: nem találhatók pontpárok.": "ROI matching: no point pairs found.",
    "ROI illesztés megszakítva.": "ROI matching cancelled.",
    "ROI illesztés sikertelen.": "ROI matching failed.",

    "Kép {slot} betöltve: {Path(path).name}  ": "Image {slot} loaded: {Path(path).name}  ",
    "Kép A+B betöltve  –  {len(imgs) - 2} fájl figyelmen kívül hagyva.": "Images A+B loaded  –  {len(imgs) - 2} files ignored.",

    "Projekt mentve: {Path(path).name}": "Project saved: {Path(path).name}",
    "Projekt betöltve: {Path(path).name}  ": "Project loaded: {Path(path).name}  ",
    "{total} képkocka  |  ~{total / self._spin_fps.value():.1f} s": "{total} frames  |  ~{total / self._spin_fps.value():.1f} s",

    "GIF elmentve: {Path(path).name}": "GIF saved: {Path(path).name}",
    "{len(self._frames)} PNG elmentve → {Path(folder).name}/": "{len(self._frames)} PNGs saved → {Path(folder).name}/",

    # Message Boxes
    "Hiányzó kép": "Missing Image",
    "Tölts be mindkét képet!": "Load both images!",

    "Hiba a kép betöltésekor": "Error Loading Image",
    "Hiba az illesztés során:": "Error during matching:",

    "Illesztési hiba": "Matching Error",
    "ROI illesztési hiba": "ROI Matching Error",

    "Generálási hiba": "Generation Error",
    "Nincs adat az összehasonlításhoz.": "No data to compare.",
    "Nincs előnézet.": "No preview.",
    "Nincs exportálható képkocka.": "No frames to export.",

    "Mentési hiba": "Save Error",
    "Betöltési hiba": "Load Error",

    "Megerősítés": "Confirmation",
    "Biztosan törlöd az összes pontpárt?": "Delete all point pairs?",
    "Összes pontpár törölve.": "All point pairs deleted.",

    "Profil": "Profile",
    "Add meg a profil nevét!": "Enter profile name!",
    "Érvénytelen profil név.": "Invalid profile name.",
    "Nem található: {name}.toml": "Not found: {name}.toml",

    "Profil törlése": "Delete Profile",
    "Visszaállítás": "Reset",
    "Minden beállítást visszaállít az alapértelmezett értékre.\n\nBiztosan?": "Reset all settings to defaults?\n\nAre you sure?",

    "Backend nem elérhető": "Backend Not Available",

    # ══════════════════════════════════════════════════════════════════════════
    # ERROR AND WARNING MESSAGES
    # ══════════════════════════════════════════════════════════════════════════

    "Az OpenCV (cv2) nincs telepítve.": "OpenCV (cv2) is not installed.",
    "Az OpenCV (cv2) nincs telepítve.": "OpenCV (cv2) is not installed.",
    "A PyTorch nincs telepítve.": "PyTorch is not installed.",
    "A lightglue csomag nincs telepítve.": "lightglue package is not installed.",
    "A DISK extractor nincs elérhető (lightglue >= 0.1 szükséges).": "DISK extractor not available (lightglue >= 0.1 required).",
    "A kornia nincs telepítve.\nTelepítés: pip install kornia": "kornia is not installed.\nInstall: pip install kornia",

    "A Pillow könyvtár nincs telepítve.\nTelepítsd: pip install Pillow": "Pillow is not installed.\nInstall: pip install Pillow",

    "Nem sikerült megnyitni a videó-írót: {path}": "Failed to open video writer: {path}",
    "Nincs exportálható képkocka.": "No frames to export.",

    "A ROI kívül esik a képen vagy üres.": "ROI is outside image or empty.",
    "Delaunay morphhoz legalább 3 pontpár szükséges.\nAdj hozzá több anchor pontot, vagy válassz más módszert!": "Delaunay morph requires at least 3 point pairs.\nAdd more anchor points or choose another method!",

    # ══════════════════════════════════════════════════════════════════════════
    # HELP TEXT AND DESCRIPTIONS
    # ══════════════════════════════════════════════════════════════════════════

    # SuperPoint Help Text
    "A SuperPoint detektor által megtartott kulcspontok maximális száma.\nTöbb pont → pontosabb illesztés, de lassabb futás és több GPU-memória.\n\nForrás: a cvg/LightGlue GitHub-repo hivatalos demói és benchmarkjai\nAjánlott: 2048–4096. Tartomány: 128–8192.": "Maximum keypoints kept by SuperPoint detector.\nMore points → more accurate matching, but slower and more GPU memory.\n\nSource: official demos and benchmarks from cvg/LightGlue GitHub repo\nRecommended: 2048–4096. Range: 128–8192.",

    "A SuperPoint detektor érzékenységi küszöbe (detection_threshold).\nAlacsonyabb érték → több pont detektálódik, beleértve a gyengébbeket is.\nMagasabb érték → csak az erős, megbízható sarokpontok maradnak meg.\n\nForrás: SuperPoint hivatalos implementáció (Magic Leap / rpautrat/SuperPoint),\nHugging Face transformers, cvg/LightGlue – mindegyik 0.005-öt használ.\nAjánlott: 0.001–0.010.": "SuperPoint detection threshold (detection_threshold).\nLower → more points detected, including weaker ones.\nHigher → only strong, reliable corner points remain.\n\nSource: official SuperPoint implementation (Magic Leap/rpautrat),\nHugging Face transformers, cvg/LightGlue all use 0.005.\nRecommended: 0.001–0.010.",

    "Non-Maximum Suppression sugár pixelben (nms_radius).\nMegakadályozza, hogy két kulcspont egymáshoz túl közel legyen.\nNagyobb érték → ritkább, de egyenletesebb eloszlású pontok.\nKisebb érték → sűrűbb pontok, de lehetséges átfedés.\n\nForrás: SuperPoint eredeti cikk (DeTone et al. 2018) és a cvg/LightGlue\nAjánlott: 3–6.": "Non-Maximum Suppression radius in pixels (nms_radius).\nPrevents two keypoints from being too close.\nLarger → sparser but evenly distributed points.\nSmaller → denser points but possible overlap.\n\nSource: SuperPoint paper (DeTone et al. 2018) and cvg/LightGlue\nRecommended: 3–6.",

    "A LightGlue párosító szűrési küszöbe (filter_threshold).\nAlacsonyabb → szigorúbb szűrés: kevesebb, de pontosabb egyezés.\nMagasabb → több egyezés, de több hibás pár is átcsúszik a szűrőn.\n\nForrás: cvg/LightGlue – lightglue/lightglue.py, 'filter_threshold: 0.1'\nEz a könyvtár hivatalos alapértelmezettje.\nAjánlott: 0.05–0.20.": "LightGlue matching filter threshold (filter_threshold).\nLower → stricter filtering: fewer but more accurate matches.\nHigher → more matches, but more false pairs slip through.\n\nSource: cvg/LightGlue – lightglue/lightglue.py, 'filter_threshold: 0.1'\nThis is the library's official default.\nRecommended: 0.05–0.20.",

    "A LightGlue korai leállás küszöbe mélység irányban (depth_confidence).\nHa az egyezések elég megbízhatók, a modell korábban megáll → gyorsabb.\n\nForrás: LightGlue ICCV 2023 cikk (Lindenberger et al.) és a cvg/LightGlue\nAjánlott: 0.90–0.98.": "LightGlue early stopping threshold in depth direction (depth_confidence).\nIf matches are confident enough, the model stops early → faster.\n\nSource: LightGlue ICCV 2023 paper (Lindenberger et al.) and cvg/LightGlue\nRecommended: 0.90–0.98.",

    "A LightGlue korai leállás küszöbe szélesség irányban (width_confidence).\nHa az összes pont kezelve van elég biztosan, a modell megáll → gyorsabb.\n\nForrás: LightGlue ICCV 2023 cikk és a cvg/LightGlue repo –\nAjánlott: 0.95–1.0.": "LightGlue early stopping threshold in width direction (width_confidence).\nIf all points are handled confidently enough, the model stops → faster.\n\nSource: LightGlue ICCV 2023 paper and cvg/LightGlue repo\nRecommended: 0.95–1.0.",

    # DISK Help Text
    "A DISK detektor által megtartott kulcspontok maximális száma.\nA DISK ismétlődő mintákon (ablaksorok, csempék, kövezet) jobban\nteljesít mint a SuperPoint, mert ezekre van betanítva.\n\nForrás: Parskatt/DeDoDe és a kornia.feature.DISK implementációk\nAjánlott: 3000–8000.": "Maximum keypoints kept by DISK detector.\nDISK performs better on repetitive patterns (window grids, tiles, pavement)\nbecause it's trained for them.\n\nSource: Parskatt/DeDoDe and kornia.feature.DISK implementations\nRecommended: 3000–8000.",

    # LoFTR Help Text
    "Az előre tanított LoFTR modell típusa.\n\nVálaszd a képeid tartalmához leginkább illőt.": "Type of pretrained LoFTR model.\n\nChoose the one that best matches your image content.",

    "Az egyezés megbízhatósági küszöbe.\nCsak az ennél magasabb konfidenciájú egyezések maradnak meg.\nAlacsonyabb → több egyezés, de több zajpont is.\nMagasabb → kevesebb, de pontosabb egyezés.\nAjánlott: 0.30–0.70.": "Match confidence threshold.\nOnly matches with higher confidence are kept.\nLower → more matches but more noise.\nHigher → fewer but more accurate matches.\nRecommended: 0.30–0.70.",

    # SIFT Help Text
    "A SIFT által detektált kulcspontok maximális száma (nfeatures).\n\nForrás: Lowe IJCV 2004 eredeti cikkje és az OpenCV SIFT dokumentáció –\nHa sok a felesleges pont, a RANSAC szűrő eltávolítja őket.\nAjánlott: 0 (korlátlan) vagy 2000–5000 ha lassú a feldolgozás.": "Maximum keypoints detected by SIFT (nfeatures).\n\nSource: Lowe's original IJCV 2004 paper and OpenCV SIFT documentation\nIf there are too many extra points, RANSAC filter removes them.\nRecommended: 0 (unlimited) or 2000–5000 if processing is slow.",

    "Az oktávonkénti rétegek száma a Gauss-skálatérben (nOctaveLayers).\nTöbb réteg → finomabb méretarány-invariancia, de lassabb futás.\n\nForrás: Lowe IJCV 2004 – 3 az eredeti értéke ('s=3 gives good results').\nAz OpenCV alapértelmezettje is 3. Általában nem érdemes változtatni.": "Number of layers per octave in the Gaussian scale-space (nOctaveLayers).\nMore layers → finer scale invariance, but slower execution.\n\nSource: Lowe IJCV 2004 – original value is 3 ('s=3 gives good results').\nOpenCV default is also 3. Generally not worth changing.",

    "Alacsony kontrasztú kulcspontok szűrési küszöbe (contrastThreshold).\nAlacsonyabb → több pont, beleértve a gyenge kontrasztú területeket.\nMagasabb → csak erős kontrasztú, megbízható pontok maradnak.\n\nForrás: Lowe IJCV 2004 – 0.04 az eredeti ajánlott érték.\nAz OpenCV alapértelmezettje szintén 0.04.\nAjánlott: 0.02–0.08.": "Low-contrast keypoint filtering threshold (contrastThreshold).\nLower → more points, including low-contrast areas.\nHigher → only strong contrast, reliable points remain.\n\nSource: Lowe IJCV 2004 – 0.04 is the original recommended value.\nOpenCV default is also 0.04.\nRecommended: 0.02–0.08.",

    "Az élszűrő küszöbe (edgeThreshold).\nAz élek mentén lévő instabil kulcspontokat szűri ki.\nMagasabb → kevesebb él-jellegű pont törlése (több pont marad).\nAlacsonyabb → szigorúbb élszűrés (kevesebb, de stabilabb pont).\n\nForrás: Lowe IJCV 2004 – 10 az eredeti érték ('r=10').\nAz OpenCV alapértelmezettje szintén 10.\nAjánlott: 5–20.": "Edge filter threshold (edgeThreshold).\nFilters out unstable keypoints along edges.\nHigher → less edge point removal (more points remain).\nLower → stricter edge filtering (fewer but more stable points).\n\nSource: Lowe IJCV 2004 – 10 is the original value ('r=10').\nOpenCV default is also 10.\nRecommended: 5–20.",

    "A Gauss-simítás sigma paramétere a skálatér alaplépésénél.\nKisebb (1.2–1.4): kis képeken, kevésbé zajos képeken.\nNagyobb (1.8–2.5): nagy, zajos képeken, erősebb simítás.\n\nForrás: Lowe IJCV 2004 – 1.6 az eredeti értéke ('σ=1.6').\nAz OpenCV alapértelmezettje szintén 1.6. Általában nem kell változtatni.": "Gaussian smoothing sigma parameter at the scale-space base step.\nSmaller (1.2–1.4): on small, less noisy images.\nLarger (1.8–2.5): on large, noisy images, stronger smoothing.\n\nSource: Lowe IJCV 2004 – original value is 1.6 ('σ=1.6').\nOpenCV default is also 1.6. Generally no need to change.",

    "Lowe-féle arányküszöb a hamis egyezések szűrésére.\nEgy egyezés elfogadott, ha:\nlegjobb_távolság < arány × második_legjobb_távolság\nAlacsonyabb → szigorúbb: kevesebb, de pontosabb egyezés.\nMagasabb → több egyezés, de több hamis pár is.\n\nForrás: Lowe IJCV 2004 – 0.8 az eredeti mért optimális érték\nAjánlott: 0.70–0.85.": "Lowe's ratio threshold for filtering false matches.\nA match is accepted if:\nbest_distance < ratio × second_best_distance\nLower → stricter: fewer but more accurate matches.\nHigher → more matches but more false pairs.\n\nSource: Lowe IJCV 2004 – 0.8 is the original optimal value\nRecommended: 0.70–0.85.",

    # RANSAC Help Text
    "RANSAC (Random Sample Consensus) geometriai szűrés.\nAutomatikusan kiszűri a geometriailag hibás párokat (outliereket)\naz optikai adatok alapján.\nErősen ajánlott, különösen sok pont és eltérő perspektíva esetén.": "RANSAC (Random Sample Consensus) geometric filtering.\nAutomatically filters geometrically incorrect pairs (outliers)\nbased on optical data.\nStrongly recommended, especially with many points and different perspectives.",

    "A RANSAC visszavetítési (reprojekciós) küszöb pixelben.\nHa egy pontpár visszavetítési hibája nagyobb ennél, outliernek\nminősül és törlődik az illesztésből.\nKisebb → szigorúbb szűrés (kevesebb pont, de pontosabb).\nNagyobb → engedékenyebb szűrés (több pont marad).\n\nForrás: OpenCV cv2.findHomography dokumentáció –\nOpenCV feature homography tutorial szintén 3.0-t ajánl.\nAjánlott: 2.0–5.0. Pixelszintű pontossághoz: 1.0–2.0.": "RANSAC reprojection error threshold in pixels.\nIf a point pair's reprojection error is larger, it's considered an outlier\nand removed from matching.\nSmaller → stricter filtering (fewer points but more accurate).\nLarger → more lenient filtering (more points remain).\n\nSource: OpenCV cv2.findHomography documentation\nOpenCV feature homography tutorial also recommends 3.0.\nRecommended: 2.0–5.0. For pixel accuracy: 1.0–2.0.",

    # ══════════════════════════════════════════════════════════════════════════
    # POINT EDITOR SPECIFIC MESSAGES
    # ══════════════════════════════════════════════════════════════════════════
    "Ctrl+klikk: csúcs ({n} db)  |  Enter / Ctrl+2×klikk: lezár": "Ctrl+click: vertex ({n} count)  |  Enter / Ctrl+2×click: close",

    # ══════════════════════════════════════════════════════════════════════════
    # MISC UI ELEMENTS
    # ══════════════════════════════════════════════════════════════════════════
    "Lejátszási sebesség (képkocka/másodperc)": "Playback speed (frames per second)",

    # Export Format Info
    "Delaunay háromszög: klasszikus face-morph, legjobb minőség (pontpár kell)\nOptikai folyam: dense flow alapú, organikus mozgás (pontpár nélkül is)\nHomográfia: perspektíva-interpoláció, gyors (homográfia mátrix kell)": "Delaunay Triangulation: classic face-morph, best quality (needs point pairs)\nOptical Flow: dense flow based, organic movement (works without pairs)\nHomography: perspective interpolation, fast (needs homography matrix)",

    "Lineáris: egyenletes tempó\nLassú start: kezdet lassú, majd gyorsít\nLassú vége: gyors kezdet, majd lelassít\nS-görbe: lassú start + lassú vég (ajánlott)": "Linear: constant tempo\nEase In: slow start, then speeds up\nEase Out: fast start, then slows down\nS-Curve: slow start + slow end (recommended)",

    # Dialog file type filters (typically not translated but included for completeness)
    "ArchMorph projekt (*.json)": "ArchMorph Project (*.json)",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Language engine
# ══════════════════════════════════════════════════════════════════════════════

_current_lang: str = "hu"   # default: Hungarian


def set_language(lang: str) -> None:
    """Set active language.  Supported: 'hu' (default), 'en'."""
    global _current_lang
    if lang in ("hu", "en"):
        _current_lang = lang


def get_language() -> str:
    """Return the currently active language code."""
    return _current_lang


def tr(text: str) -> str:
    """Translate *text* to the active language.

    When language is 'hu' (default) the original Hungarian text is returned
    unchanged.  When 'en', the English translation is looked up; if no entry
    exists the original text is returned as fallback so nothing ever breaks.
    """
    if _current_lang == "en":
        return _EN.get(text, text)
    return text
