# 🔬 Scintigraphy Blob Tracker

> Tracking de silhouette en temps réel avec esthétique scintigraphie / CRT sci-fi, pensé pour les performances live et le VJing.

![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-informational) ![Python](https://img.shields.io/badge/Python-OpenCV%20%2B%20Pygame-blue) ![Style](https://img.shields.io/badge/Style-Medical%20Scifi-00ff88)

## À propos

Application Python qui capte une webcam (ou un fichier vidéo), isole la silhouette par soustraction de fond, puis la rend comme une image de scintigraphie médicale : palette froide (noir → bleu foncé → cyan → blanc), traînées, glow, grain et effets d'écran CRT. Les paramètres se règlent en direct dans une fenêtre de contrôle séparée, ce qui laisse la fenêtre de rendu propre pour la projection ou la capture.

## Fonctionnalités

- **Entrée webcam ou fichier vidéo** (la vidéo boucle automatiquement)
- **Soustraction de fond MOG2** (OpenCV) et détection de blob (contour + centroïde)
- **Traînées** avec intensité et décroissance réglables, **particules** optionnelles
- **Mode Scanner** : révélation progressive façon radiographie (touche `S`), vitesse réglable
- **Effets visuels** : glow, effet de profondeur, grain de film, scanlines, aberration chromatique, luminosité / contraste
- **Squelette MediaPipe** (optionnel) superposé, avec opacité réglable — activé automatiquement si MediaPipe est installé et compatible
- **Fenêtre de contrôle séparée** (« Scintigraphy Controls ») avec curseurs, affichage des FPS, du mode et de l'état du squelette

### Pistes d'évolution

Idées notées dans l'ancienne documentation : export des paramètres en preset JSON, sortie OSC (Resolume/TouchDesigner), tracking multi-personne, enregistrement/replay de timeline, shaders GLSL personnalisés, sortie NDI intégrée.

## Installation

Prérequis : **Python 3.8+**, une **webcam** (ou un fichier vidéo).

### Lancement rapide

Les scripts créent un environnement virtuel `venv` au premier lancement, installent les dépendances puis lancent l'app :

- **Windows** : double-clic sur `start_tracker.bat`
- **Linux / macOS** :
  ```bash
  chmod +x start_tracker.sh   # si « permission denied »
  ./start_tracker.sh
  ```

Guide pas à pas : [QUICK_START.md](QUICK_START.md).

### Installation manuelle

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac :**
```bash
python3 -m venv venv
source venv/bin/activate
```

Puis :
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Utilisation

```bash
python scintigraphy_tracker.py              # Webcam
python scintigraphy_tracker.py video.mp4    # Fichier vidéo
```

### Contrôles

| Touche | Action |
|--------|--------|
| **S** | Activer/désactiver le mode Scanner |
| **Échap** | Quitter |

### Paramètres (fenêtre « Scintigraphy Controls »)

`trail_intensity`, `trail_decay`, `particle_emission` (désactivé par défaut), `glow_strength`, `grain_intensity`, `depth_intensity`, `scanline_intensity`, `chromatic_aberration`, `brightness`, `contrast`, `scanner_speed`, et `skeleton_opacity` quand le squelette MediaPipe est actif.

### Conseils pour la performance

**Calibration :**
1. Lance l'app dans une pièce bien éclairée
2. Place-toi face à la webcam (2-3 mètres)
3. Laisse quelques secondes à la soustraction de fond (MOG2) pour se stabiliser
4. Bouge lentement pour tester le tracking

**Éclairage recommandé :**
- **Lumière directe** sur toi (pas de contre-jour)
- **Fond uni** si possible (mur blanc/noir)
- Évite les **ombres fortes** ou **reflets** ; des **vêtements contrastés** rendent la silhouette plus visible

### Intégration dans un workflow VJ

- **Capture avec OBS** : Source → **Window Capture** → « Scintigraphy Blob Tracker », puis enregistrer.
- **Vers Resolume** : passer par OBS Virtual Camera, Spout ou NDI.
- **En live** : fenêtre de rendu sur un écran dédié / projecteur / LED wall, la fenêtre de contrôle restant sur l'écran de régie.

### Dépannage

- **Webcam non détectée** : vérifie qu'elle n'est pas utilisée par une autre app (Teams, OBS…). Sinon, dans `scintigraphy_tracker.py`, change l'index de `cv2.VideoCapture(0)` (essaye 1, 2, 3).
- **FPS bas** : baisse la résolution passée à `ScintigraphyTracker(width=1280, height=720, …)` dans `main()` (par ex. `width=960, height=540`) et laisse `particle_emission` à 0.
- **« No module named 'cv2' »** : les dépendances ne sont pas installées → `pip install -r requirements.txt`.
- **Test rapide de la webcam** : `python -c "import cv2; cv2.VideoCapture(0).read()"`

### Pipeline de rendu

```
Frame webcam / vidéo (BGR)
    ↓
Soustraction de fond (MOG2) → masque
    ↓
Détection de blob (OpenCV) → contour + centroïde
    ↓
Traînées (accumulation + decay) · particules · squelette MediaPipe (optionnel)
    ↓
Palette scintigraphie (LUT froide) · mode Scanner
    ↓
Effets : profondeur, glow, grain, aberration chromatique, scanlines
    ↓
Rendu Pygame
```

## Structure du projet

```
.
├── scintigraphy_tracker.py  # Application principale
├── requirements.txt         # Dépendances Python
├── start_tracker.bat        # Installation + lancement (Windows)
├── start_tracker.sh         # Installation + lancement (Linux/macOS)
├── config_example.py        # Exemples de presets, palettes et réglages caméra/affichage
└── QUICK_START.md           # Guide d'installation rapide
```

`config_example.py` regroupe des presets (`medical_cold`, `radioactive_intense`, `crt_retro`, `ghost_minimal`, `performance_mode`) et des palettes ; il n'est pas encore lu par `scintigraphy_tracker.py`.

## Stack technique

Python, OpenCV, NumPy, Pygame, MediaPipe (optionnel), Pillow.

## Crédits & licence

Aucun fichier de licence n'est présent. L'ancienne documentation indiquait : « Code libre d'utilisation pour tes performances live ! »

---

<p align="center">
  Fait par <a href="https://github.com/elisalien">Élisalien</a> ·
  <a href="https://gitlab.com/elisalien/catalog">Catalogue des shaders, sources & logiciels</a>
</p>
