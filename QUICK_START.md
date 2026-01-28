# 🚀 INSTALLATION RAPIDE (5 MINUTES)

Guide simplifié pour lancer l'app sans galère !

## ⚡ Version Express (Windows)

### 1️⃣ Installe Python (si pas déjà fait)

Télécharge depuis : https://www.python.org/downloads/

⚠️ **IMPORTANT** : Coche "Add Python to PATH" pendant l'installation !

### 2️⃣ Vérifie CUDA

Ouvre PowerShell et tape :
```bash
nvidia-smi
```

Tu devrais voir ta RTX 3080 Ti. Si erreur → installe CUDA Toolkit :
https://developer.nvidia.com/cuda-downloads

### 3️⃣ Télécharge les fichiers

Décompresse le dossier `scintigraphy-tracker` sur ton Bureau.

### 4️⃣ Lance l'app

**Double-clic sur `start_tracker.bat`**

C'est tout ! La première fois ça va :
- Créer un environnement Python isolé
- Télécharger les librairies (5-10 min)
- Télécharger le modèle IA (~400 MB)
- Lancer l'app automatiquement

Les fois suivantes, ça lance direct en 2 secondes !

---

## 🐧 Version Express (Linux/Mac)

### 1️⃣ Ouvre un Terminal

Va dans le dossier de l'app :
```bash
cd ~/Downloads/scintigraphy-tracker  # Adapte le chemin
```

### 2️⃣ Lance le script

```bash
./start_tracker.sh
```

Si erreur "permission denied" :
```bash
chmod +x start_tracker.sh
./start_tracker.sh
```

---

## 🎮 Premiers pas

### Au démarrage de l'app

1. **Autorise l'accès webcam** si demandé
2. **Positionne-toi** face à la caméra (2-3 mètres)
3. **Attends 5 secondes** que l'algo se calibre
4. **Bouge lentement** pour voir le tracking

### Contrôles de base

- **Flèches ↑↓** : Changer de paramètre
- **Flèches ←→** : Ajuster le paramètre sélectionné
- **H** : Masquer l'interface (pour les screenshots/vidéos)
- **R** : Reset tous les paramètres
- **ESC** : Quitter

### Premiers réglages à tester

1. Commence par ajuster **Trail Decay** (0.90 = traînées courtes, 0.98 = traînées longues)
2. Joue avec **Particle Emission** (5 = subtil, 30 = intense)
3. Teste **Glow Radius** (10 = net, 25 = diffus)

---

## 🆘 Problèmes courants

### "Python not found"
→ Réinstalle Python et **coche "Add to PATH"**

### "No module named 'cv2'"
→ Les dépendances ne sont pas installées. Ouvre PowerShell dans le dossier :
```bash
pip install -r requirements.txt
```

### "Cannot open camera"
→ Vérifie que ta webcam n'est pas utilisée par une autre app (Teams, OBS, etc.)

### "CUDA error"
→ Installe CUDA Toolkit depuis NVIDIA. L'app fonctionne quand même en mode CPU (plus lent).

### FPS trop bas
→ Baisse la résolution : ouvre `scintigraphy_tracker.py` et change ligne 28 :
```python
tracker = ScintigraphyTracker(width=960, height=540)
```

---

## 📊 Utilisation en performance live

### Setup recommandé

```
PC avec app
    ↓
HDMI → Projecteur/LED
    OU
OBS → Virtual Camera → Resolume
```

### Pour capturer avec OBS

1. Source → **Window Capture**
2. Sélectionne "Scintigraphy Blob Tracker"
3. Appuie sur **H** dans l'app pour masquer l'interface
4. Record !

### Presets rapides

**Médical froid :**
- Trail Decay: 0.93
- Particle Emission: 12
- Glow: 15
- Chromatic Aberration: 1.5

**Radioactif intense :**
- Trail Decay: 0.97
- Particle Emission: 25
- Glow: 22
- Chromatic Aberration: 3.5

**CRT rétro :**
- Scanline Intensity: 0.5
- Phosphor Persistence: 0.95
- Noise: 0.12
- Gamma: 0.9

---

## 💡 Tips

- **Fond uni** = meilleur tracking
- **Éclairage direct** sur toi (pas de contre-jour)
- **Vêtements contrastés** = plus visible
- **Mouvements lents** au début pour tester

---

**C'est parti ! 🔬✨**

Si ça marche pas, check le README.md complet pour plus de détails.
