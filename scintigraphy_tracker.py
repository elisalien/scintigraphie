"""
Scintigraphy-Inspired Blob Tracker with CRT Effects
Real-time webcam tracking with medical/scifi aesthetic
+ MediaPipe Skeleton Tracking (optional)
+ Scanner Mode (radiographic reveal effect)
+ Video file input support
+ Separate Control Window

Usage:
  python scintigraphy_tracker.py              # Use webcam
  python scintigraphy_tracker.py video.mp4    # Use video file
"""

import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List
import sys
import os

# Try to import pygame
try:
    import pygame
    from pygame.locals import *
    PYGAME_AVAILABLE = True
except ImportError:
    print("❌ pygame not installed. Run: pip install pygame")
    PYGAME_AVAILABLE = False

# Try to import MediaPipe for skeleton tracking
MEDIAPIPE_AVAILABLE = False
mp = None
mp_pose = None
mp_drawing = None

try:
    import mediapipe
    if hasattr(mediapipe, 'solutions'):
        mp = mediapipe
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        MEDIAPIPE_AVAILABLE = True
        print("✓ MediaPipe skeleton tracking available")
    else:
        print("⚠️  MediaPipe installed but API incompatible. Skeleton disabled.")
except ImportError:
    print("⚠️  MediaPipe not installed. Skeleton tracking disabled.")
except Exception as e:
    print(f"⚠️  MediaPipe error: {e}. Skeleton tracking disabled.")


@dataclass
class Particle:
    """Radioactive trail particle"""
    x: float
    y: float
    vx: float
    vy: float
    life: float
    intensity: float
    size: float


class ControlWindow:
    """Separate OpenCV window for parameter controls"""

    def __init__(self, params, has_skeleton=False):
        self.params = params
        self.window_name = "Scintigraphy Controls"
        self.has_skeleton = has_skeleton

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 550)

        self.trackbar_config = {
            'trail_intensity': (0, 100, 85),
            'trail_decay': (0, 100, 92),
            'particle_emission': (0, 20, 0),  # Disabled by default
            'glow_strength': (0, 100, 40),
            'grain_intensity': (0, 100, 35),  # Film grain effect
            'depth_intensity': (0, 100, 50),  # Depth map effect
            'scanline_intensity': (0, 50, 8),
            'chromatic_aberration': (0, 10, 1),
            'brightness': (50, 200, 100),
            'contrast': (50, 200, 115),
            'scanner_speed': (1, 20, 5),
        }

        if has_skeleton:
            self.trackbar_config['skeleton_opacity'] = (0, 100, 70)

        for name, (min_val, max_val, default) in self.trackbar_config.items():
            cv2.createTrackbar(name, self.window_name, default, max_val, lambda x: None)

    def update_params(self):
        """Read trackbar values"""
        try:
            self.params['trail_intensity'] = cv2.getTrackbarPos('trail_intensity', self.window_name) / 100.0
            self.params['trail_decay'] = cv2.getTrackbarPos('trail_decay', self.window_name) / 100.0
            self.params['particle_emission'] = cv2.getTrackbarPos('particle_emission', self.window_name)
            self.params['glow_strength'] = cv2.getTrackbarPos('glow_strength', self.window_name) / 100.0
            self.params['grain_intensity'] = cv2.getTrackbarPos('grain_intensity', self.window_name) / 100.0
            self.params['depth_intensity'] = cv2.getTrackbarPos('depth_intensity', self.window_name) / 100.0
            self.params['scanline_intensity'] = cv2.getTrackbarPos('scanline_intensity', self.window_name) / 100.0
            self.params['chromatic_aberration'] = cv2.getTrackbarPos('chromatic_aberration', self.window_name)
            self.params['brightness'] = cv2.getTrackbarPos('brightness', self.window_name) / 100.0
            self.params['contrast'] = cv2.getTrackbarPos('contrast', self.window_name) / 100.0
            self.params['scanner_speed'] = cv2.getTrackbarPos('scanner_speed', self.window_name)

            if self.has_skeleton:
                self.params['skeleton_opacity'] = cv2.getTrackbarPos('skeleton_opacity', self.window_name) / 100.0
        except cv2.error:
            pass

    def show(self, fps, mode, skeleton_active=False):
        """Display control panel"""
        info = np.zeros((550, 400, 3), dtype=np.uint8)
        info[:] = (30, 30, 30)

        cv2.putText(info, "SCINTIGRAPHY CONTROLS", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 150), 2)  # Cyan

        fps_color = (255, 200, 100) if fps > 30 else (100, 100, 255)  # Cyan/Red
        cv2.putText(info, f"FPS: {fps:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1)

        # Mode display
        mode_color = (255, 255, 200) if mode == "SCANNER" else (255, 180, 100)  # Bright cyan
        cv2.putText(info, f"Mode: {mode}", (200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)

        if self.has_skeleton:
            status = "ACTIVE" if skeleton_active else "No body"
            status_color = (255, 220, 150) if skeleton_active else (100, 100, 100)  # Cyan
            cv2.putText(info, f"Skeleton: {status}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

        # Keyboard shortcuts
        cv2.putText(info, "KEYBOARD SHORTCUTS:", (20, 480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(info, "S = Toggle Scanner Mode", (20, 505),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(info, "ESC = Quit", (20, 525),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow(self.window_name, info)


class ScintigraphyTracker:
    def __init__(self, width=1280, height=720, video_source=None):
        self.width = width
        self.height = height
        self.video_source = video_source
        self.is_video_file = video_source is not None

        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required")

        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Scintigraphy Blob Tracker")
        self.clock = pygame.time.Clock()

        # Open video source (webcam or file)
        if self.is_video_file:
            if not os.path.exists(video_source):
                raise RuntimeError(f"❌ Video file not found: {video_source}")
            print(f"🎬 Opening video file: {video_source}")
            self.cap = cv2.VideoCapture(video_source)
        else:
            print("📹 Opening webcam...")
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open video source")

        # Set resolution for webcam, or get video dimensions
        if self.is_video_file:
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"✓ Video opened: {self.width}x{self.height} @ {self.video_fps:.1f}fps ({self.total_frames} frames)")
            # Resize pygame window to match video
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print("✓ Webcam opened")

        # MediaPipe Pose (optional)
        self.pose = None
        self.use_skeleton = False

        if MEDIAPIPE_AVAILABLE:
            try:
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmark_style = mp_drawing.DrawingSpec(
                    color=(255, 255, 200), thickness=2, circle_radius=3  # Bright cyan (BGR)
                )
                self.connection_style = mp_drawing.DrawingSpec(
                    color=(255, 200, 100), thickness=2  # Medium cyan-blue (BGR)
                )
                self.use_skeleton = True
                print("✓ Skeleton tracking initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize skeleton tracking: {e}")

        print("📊 Using MOG2 background subtraction")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # Buffers
        self.trail_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.scanner_buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Particles (reduced for performance)
        self.particles: List[Particle] = []
        self.max_particles = 300
        self.prev_centroid = None

        # Colormap LUT
        self.colormap_lut = self._create_scintigraphy_lut()

        # Parameters
        self.params = {
            'trail_intensity': 0.85,
            'trail_decay': 0.92,
            'particle_emission': 0,  # Disabled by default
            'glow_strength': 0.4,
            'grain_intensity': 0.35,  # Film grain
            'depth_intensity': 0.5,  # Depth map effect
            'scanline_intensity': 0.08,
            'chromatic_aberration': 1,
            'brightness': 1.0,
            'contrast': 1.15,
            'skeleton_opacity': 0.7,
            'scanner_speed': 5,
        }

        # Control window
        self.control_window = ControlWindow(self.params, has_skeleton=self.use_skeleton)

        # FPS tracking
        self.fps_history = deque(maxlen=30)

        # Static textures (optimized)
        self._create_static_textures()

        # Mode and scanner state
        self.scanner_mode = False
        self.scanner_y = 0
        self.skeleton_detected = False

    def _create_scintigraphy_lut(self):
        """Create authentic scintigraphy colormap lookup table

        True medical scintigraphy uses a cold palette:
        Black → Dark Blue → Cyan → White
        No greens, yellows or warm tones
        """
        lut_b = np.zeros(256, dtype=np.uint8)
        lut_g = np.zeros(256, dtype=np.uint8)
        lut_r = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            t = i / 255.0

            if t < 0.15:
                # Black to very dark blue
                r, g, b = 0, 0, int(t / 0.15 * 40)
            elif t < 0.35:
                # Dark blue to medium blue
                tt = (t - 0.15) / 0.20
                r, g, b = 0, int(tt * 30), 40 + int(tt * 120)
            elif t < 0.55:
                # Medium blue to cyan-blue
                tt = (t - 0.35) / 0.20
                r, g, b = 0, 30 + int(tt * 150), 160 + int(tt * 60)
            elif t < 0.75:
                # Cyan-blue to bright cyan
                tt = (t - 0.55) / 0.20
                r, g, b = int(tt * 80), 180 + int(tt * 55), 220 + int(tt * 35)
            elif t < 0.90:
                # Bright cyan to near-white
                tt = (t - 0.75) / 0.15
                r, g, b = 80 + int(tt * 140), 235 + int(tt * 20), 255
            else:
                # Near-white to pure white (hot spots)
                tt = (t - 0.90) / 0.10
                r, g, b = 220 + int(tt * 35), 255, 255

            lut_b[i] = min(255, b)
            lut_g[i] = min(255, g)
            lut_r[i] = min(255, r)

        return (lut_b, lut_g, lut_r)

    def _create_static_textures(self):
        """Pre-compute static textures"""
        # Scanlines (every 2nd line)
        self.scanlines = np.ones((self.height, self.width), dtype=np.float32)
        self.scanlines[1::2, :] = 0.88

        # Vignette
        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        self.vignette = (1.0 - np.clip(dist * 0.5, 0, 0.4)).astype(np.float32)

        # Pre-compute vignette as uint8 for faster blending
        self.vignette_3ch = np.dstack([self.vignette] * 3)

    def detect_pose(self, frame):
        """Detect pose using MediaPipe"""
        if not self.use_skeleton or self.pose is None:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            self.skeleton_detected = results.pose_landmarks is not None
            return results
        except:
            return None

    def draw_skeleton(self, image, pose_results):
        """Draw skeleton overlay"""
        if pose_results is None or not pose_results.pose_landmarks:
            return image

        try:
            skeleton_layer = np.zeros_like(image)
            mp_drawing.draw_landmarks(
                skeleton_layer,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_style,
                connection_drawing_spec=self.connection_style
            )

            # Simple glow
            skeleton_glow = cv2.GaussianBlur(skeleton_layer, (5, 5), 0)
            skeleton_layer = cv2.addWeighted(skeleton_layer, 1.0, skeleton_glow, 0.4, 0)

            opacity = self.params['skeleton_opacity']
            mask = cv2.cvtColor(skeleton_layer, cv2.COLOR_BGR2GRAY) > 0
            image[mask] = cv2.addWeighted(
                image, 1 - opacity,
                skeleton_layer, opacity, 0
            )[mask]
        except:
            pass

        return image

    def remove_background(self, frame):
        """Remove background"""
        return self.bg_subtractor.apply(frame)

    def detect_blob(self, mask):
        """Detect main blob"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 5000:
            return None, None

        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None

        return largest_contour, centroid

    def update_trail(self, mask):
        """Update trail buffer"""
        lut_b, lut_g, lut_r = self.colormap_lut
        b = cv2.LUT(mask, lut_b)
        g = cv2.LUT(mask, lut_g)
        r = cv2.LUT(mask, lut_r)
        colored = cv2.merge([b, g, r])

        decay = self.params['trail_decay']
        intensity = self.params['trail_intensity']

        self.trail_buffer = cv2.addWeighted(
            self.trail_buffer, decay,
            colored, intensity * (1 - decay),
            0
        )

        return self.trail_buffer

    def apply_scanner_effect(self, image):
        """Apply scanner/radiograph progressive reveal effect"""
        speed = max(1, self.params['scanner_speed'])
        self.scanner_y += speed

        if self.scanner_y >= self.height:
            self.scanner_y = 0
            self.scanner_buffer = np.zeros_like(self.scanner_buffer)

        # Copy revealed area to scanner buffer
        self.scanner_buffer[:self.scanner_y, :] = image[:self.scanner_y, :]

        # Create output with scanner effect
        output = self.scanner_buffer.copy()

        # Draw glowing scan line (cyan color scheme)
        line_y = self.scanner_y
        if line_y < self.height:
            # Main bright line (bright cyan)
            cv2.line(output, (0, line_y), (self.width, line_y), (255, 255, 200), 2)

            # Glow effect (gradient above the line)
            for i in range(1, 15):
                alpha = 1.0 - (i / 15.0)
                y = line_y - i
                if y >= 0:
                    brightness = int(200 * alpha)
                    # Cyan glow (BGR: high blue, medium green, low red)
                    cv2.line(output, (0, y), (self.width, y), (brightness, int(brightness * 0.8), int(brightness * 0.3)), 1)

            # Slight glow below
            for i in range(1, 5):
                alpha = 1.0 - (i / 5.0)
                y = line_y + i
                if y < self.height:
                    brightness = int(100 * alpha)
                    cv2.line(output, (0, y), (self.width, y), (brightness, int(brightness * 0.7), int(brightness * 0.2)), 1)

        return output

    def emit_particles(self, contour, centroid):
        """Emit particles (optimized)"""
        if contour is None or centroid is None:
            return

        velocity = (0, 0)
        if self.prev_centroid and centroid:
            velocity = (
                centroid[0] - self.prev_centroid[0],
                centroid[1] - self.prev_centroid[1]
            )

        num_particles = int(self.params['particle_emission'])
        contour_len = len(contour)

        if contour_len == 0:
            return

        for _ in range(num_particles):
            idx = np.random.randint(0, contour_len)
            point = contour[idx][0]

            dx = point[0] - centroid[0]
            dy = point[1] - centroid[1]
            dist = np.sqrt(dx**2 + dy**2) + 0.001
            dx /= dist
            dy /= dist

            particle = Particle(
                x=float(point[0]),
                y=float(point[1]),
                vx=dx * np.random.uniform(1, 2) + velocity[0] * 0.2,
                vy=dy * np.random.uniform(1, 2) + velocity[1] * 0.2,
                life=1.0,
                intensity=np.random.uniform(0.6, 1.0),
                size=np.random.uniform(2, 3)
            )
            self.particles.append(particle)

        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

        self.prev_centroid = centroid

    def update_particles(self, dt):
        """Update particles"""
        alive = []
        for p in self.particles:
            p.x += p.vx * dt * 60
            p.y += p.vy * dt * 60
            p.vx *= 0.95
            p.vy *= 0.95
            p.vy += 0.02
            p.life -= dt * 2.5

            if p.life > 0:
                alive.append(p)

        self.particles = alive

    def apply_effects_fast(self, image):
        """Apply visual effects (optimized)"""
        result = image.copy()

        # Brightness/contrast
        brightness = self.params['brightness']
        contrast = self.params['contrast']
        result = cv2.convertScaleAbs(result, alpha=contrast, beta=(brightness - 1) * 50)

        # Depth map effect - creates pseudo-3D based on intensity
        depth_intensity = self.params['depth_intensity']
        if depth_intensity > 0:
            result = self._apply_depth_effect(result, depth_intensity)

        # Film grain/noise effect
        grain_intensity = self.params['grain_intensity']
        if grain_intensity > 0:
            result = self._apply_grain(result, grain_intensity)

        # Vignette (pre-computed)
        result = (result * self.vignette_3ch).astype(np.uint8)

        # Scanlines (simple)
        if self.params['scanline_intensity'] > 0:
            scanline_mult = 1.0 - self.params['scanline_intensity'] * 0.12
            result[1::2, :] = (result[1::2, :] * scanline_mult).astype(np.uint8)

        # Chromatic aberration
        offset = int(self.params['chromatic_aberration'])
        if offset > 0:
            b, g, r = cv2.split(result)
            r = np.roll(r, offset, axis=1)
            b = np.roll(b, -offset, axis=1)
            result = cv2.merge([b, g, r])

        # Glow (small kernel for radioactive effect)
        if self.params['glow_strength'] > 0:
            glow = cv2.GaussianBlur(result, (7, 7), 0)
            result = cv2.addWeighted(result, 1.0, glow, self.params['glow_strength'] * 0.6, 0)

        return result

    def _apply_grain(self, image, intensity):
        """Apply film grain/noise effect - authentic medical imaging look"""
        # Generate noise - mix of fine and coarse grain
        h, w = image.shape[:2]

        # Fine grain (pixel-level noise)
        fine_noise = np.random.randint(-30, 30, (h, w), dtype=np.int16)

        # Coarse grain (larger splotches, more like X-ray film)
        coarse = np.random.randint(-20, 20, (h // 4, w // 4), dtype=np.int16)
        coarse_noise = cv2.resize(coarse.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.int16)

        # Combine noises
        combined_noise = (fine_noise * 0.6 + coarse_noise * 0.4).astype(np.int16)

        # Scale by intensity and image brightness (more noise in mid-tones)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Bell curve - more noise in mid-tones, less in pure black/white
        noise_mask = (4.0 * gray * (1.0 - gray))  # Peaks at 0.5

        # Apply noise
        result = image.astype(np.int16)
        for c in range(3):
            channel_noise = (combined_noise * noise_mask * intensity * 1.5).astype(np.int16)
            result[:, :, c] = np.clip(result[:, :, c] + channel_noise, 0, 255)

        return result.astype(np.uint8)

    def _apply_depth_effect(self, image, intensity):
        """Apply pseudo-depth map effect - creates 3D-like depth from intensity"""
        # Convert to grayscale to get intensity map
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create depth-based edge glow (brighter areas appear closer)
        # Use Sobel for edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = (edges / edges.max() * 255).astype(np.uint8) if edges.max() > 0 else edges.astype(np.uint8)

        # Create depth highlight (brighter = closer = more glow)
        depth_highlight = cv2.GaussianBlur(gray, (15, 15), 0)

        # Blend depth into blue channel for "depth fog" effect
        result = image.copy().astype(np.float32)

        # Add edge glow in cyan
        edge_glow = edges.astype(np.float32) / 255.0 * intensity * 0.5
        result[:, :, 0] += edge_glow * 60  # Blue
        result[:, :, 1] += edge_glow * 80  # Green (cyan tint)

        # Add depth-based intensity boost (brighter areas pop more)
        depth_boost = (depth_highlight.astype(np.float32) / 255.0) ** 1.5 * intensity * 0.3
        result[:, :, 0] += depth_boost * 40
        result[:, :, 1] += depth_boost * 50
        result[:, :, 2] += depth_boost * 30

        return np.clip(result, 0, 255).astype(np.uint8)

    def render_particles_fast(self, surface):
        """Render particles (optimized) - cyan color scheme"""
        for p in self.particles:
            if 0 <= p.x < self.width and 0 <= p.y < self.height:
                life = p.life
                intensity = p.intensity * life

                # Cyan-white color scheme (matches scintigraphy palette)
                if life > 0.6:
                    # Bright cyan-white for fresh particles
                    color = (int(200 * intensity), int(255 * intensity), int(255 * intensity))
                elif life > 0.3:
                    # Medium cyan
                    color = (int(100 * intensity), int(220 * intensity), int(255 * intensity))
                else:
                    # Fading to dark blue
                    color = (int(50 * intensity), int(150 * intensity), int(200 * intensity))

                size = max(1, int(p.size * life))
                pygame.draw.circle(surface, color, (int(p.x), int(p.y)), size)

    def handle_input(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
                if event.key == K_s:
                    self.scanner_mode = not self.scanner_mode
                    self.scanner_y = 0
                    self.scanner_buffer = np.zeros_like(self.scanner_buffer)
                    print(f"🔄 Scanner mode: {'ON' if self.scanner_mode else 'OFF'}")
        return True

    def cleanup(self):
        """Cleanup"""
        if self.pose:
            try:
                self.pose.close()
            except:
                pass
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("🔬 SCINTIGRAPHY BLOB TRACKER")
        if self.is_video_file:
            print(f"   + Video Input: {self.video_source}")
        else:
            print("   + Webcam Input")
        if self.use_skeleton:
            print("   + MediaPipe Skeleton Tracking")
        print("   + Scanner Mode (press S)")
        print("   + Separate Control Window")
        print("="*60)
        print("\n⌨️  S = Toggle Scanner | ESC = Quit")
        print("="*60 + "\n")

        running = True
        last_time = time.time()

        try:
            while running:
                current_time = time.time()
                dt = current_time - last_time
                last_time = current_time

                if dt > 0:
                    self.fps_history.append(1.0 / dt)

                running = self.handle_input()
                if not running:
                    break

                self.control_window.update_params()

                ret, frame = self.cap.read()
                if not ret:
                    if self.is_video_file:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break

                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Mirror for webcam (not for video files)
                if not self.is_video_file:
                    frame = cv2.flip(frame, 1)

                # Pose detection
                pose_results = self.detect_pose(frame)

                # Background removal
                mask = self.remove_background(frame)

                # Blob detection
                contour, centroid = self.detect_blob(mask)

                # Trail
                colored_trail = self.update_trail(mask)

                # Particles
                if contour is not None:
                    self.emit_particles(contour, centroid)
                self.update_particles(dt)

                # Effects
                final_image = self.apply_effects_fast(colored_trail)

                # Scanner mode
                if self.scanner_mode:
                    final_image = self.apply_scanner_effect(final_image)

                # Skeleton overlay
                if self.use_skeleton:
                    final_image = self.draw_skeleton(final_image, pose_results)

                # Convert to Pygame
                final_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                final_rgb = np.rot90(final_rgb)
                final_rgb = np.flipud(final_rgb)
                pygame_surface = pygame.surfarray.make_surface(final_rgb)

                self.screen.blit(pygame_surface, (0, 0))
                self.render_particles_fast(self.screen)
                pygame.display.flip()

                # Control window
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                mode = "SCANNER" if self.scanner_mode else "NORMAL"
                self.control_window.show(avg_fps, mode, self.skeleton_detected)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False
                elif key == ord('s'):
                    self.scanner_mode = not self.scanner_mode
                    self.scanner_y = 0

                self.clock.tick(60)

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
            print("\n✨ Scintigraphy Tracker closed.")


def main():
    print("\n" + "="*60)
    print("🔬 SCINTIGRAPHY BLOB TRACKER")
    print("   Medical Scifi Real-time Performance Tool")
    print("="*60 + "\n")

    # Check for video file argument
    video_source = None
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        print(f"📁 Video input: {video_source}")
    else:
        print("📹 Using webcam (pass video path as argument to use video file)")

    try:
        tracker = ScintigraphyTracker(width=1280, height=720, video_source=video_source)
        tracker.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
