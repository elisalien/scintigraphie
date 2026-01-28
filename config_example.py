# PRESETS CONFIGURATION
# Copy this file to 'config.py' and customize your favorite settings

PRESETS = {
    'medical_cold': {
        'heatmap_intensity': 2.0,
        'trail_decay': 0.93,
        'particle_emission': 12,
        'glow_radius': 15,
        'scanline_intensity': 0.25,
        'chromatic_aberration': 1.5,
        'phosphor_persistence': 0.75,
        'noise_level': 0.04,
        'gamma': 1.3,
        'contrast': 1.2,
    },
    
    'radioactive_intense': {
        'heatmap_intensity': 4.5,
        'trail_decay': 0.97,
        'particle_emission': 25,
        'glow_radius': 22,
        'scanline_intensity': 0.2,
        'chromatic_aberration': 3.5,
        'phosphor_persistence': 0.9,
        'noise_level': 0.06,
        'gamma': 1.1,
        'contrast': 1.4,
    },
    
    'crt_retro': {
        'heatmap_intensity': 2.8,
        'trail_decay': 0.94,
        'particle_emission': 15,
        'glow_radius': 18,
        'scanline_intensity': 0.5,
        'chromatic_aberration': 2.5,
        'phosphor_persistence': 0.95,
        'noise_level': 0.12,
        'gamma': 0.9,
        'contrast': 1.5,
    },
    
    'ghost_minimal': {
        'heatmap_intensity': 1.5,
        'trail_decay': 0.98,
        'particle_emission': 8,
        'glow_radius': 20,
        'scanline_intensity': 0.15,
        'chromatic_aberration': 1.0,
        'phosphor_persistence': 0.85,
        'noise_level': 0.03,
        'gamma': 1.4,
        'contrast': 1.1,
    },
    
    'performance_mode': {
        # Optimisé pour performance (FPS élevé)
        'heatmap_intensity': 2.5,
        'trail_decay': 0.90,  # Moins de traînées = plus rapide
        'particle_emission': 8,  # Moins de particules
        'glow_radius': 10,  # Moins de blur
        'scanline_intensity': 0.2,
        'chromatic_aberration': 1.0,
        'phosphor_persistence': 0.6,
        'noise_level': 0.02,
        'gamma': 1.2,
        'contrast': 1.3,
    },
}

# COLOR SCHEMES (RGB values)
# Modify these to change the heatmap colors

COLOR_SCHEMES = {
    'medical': {
        # Blue -> Cyan -> Green -> Yellow -> Red (default)
        'cold': (255, 100, 0),    # Blue (BGR format)
        'warm': (0, 100, 255),    # Red
    },
    
    'neon': {
        # Cyan -> Magenta -> Yellow
        'cold': (255, 255, 0),    # Cyan
        'warm': (255, 0, 255),    # Magenta
    },
    
    'fire': {
        # Black -> Red -> Orange -> Yellow -> White
        'cold': (0, 0, 50),       # Dark red
        'warm': (0, 150, 255),    # Orange-yellow
    },
    
    'ice': {
        # Dark blue -> Cyan -> White
        'cold': (255, 50, 0),     # Deep blue
        'warm': (255, 255, 255),  # White
    },
    
    'matrix': {
        # Black -> Dark green -> Bright green
        'cold': (0, 20, 0),       # Almost black
        'warm': (0, 255, 0),      # Bright green
    },
    
    'vaporwave': {
        # Pink -> Purple -> Cyan
        'cold': (255, 100, 180),  # Pink
        'warm': (255, 255, 50),   # Cyan
    },
}

# CAMERA SETTINGS
CAMERA_INDEX = 0  # Change if your webcam is not on index 0 (try 1, 2, 3)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# DISPLAY SETTINGS
FULLSCREEN = False  # Set to True for fullscreen mode
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# PERFORMANCE SETTINGS
TARGET_FPS = 60
ENABLE_GPU = True  # GPU-accelerated background removal

# DEBUG
SHOW_FPS = True
SHOW_UI_ON_START = True
VERBOSE_LOGGING = False
