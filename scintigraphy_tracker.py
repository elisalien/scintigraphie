"""
Scintigraphy-Inspired Blob Tracker with CRT Effects
Real-time webcam tracking with medical/scifi aesthetic
+ MediaPipe Skeleton Tracking (optional)
+ Separate Control Window
"""

import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Optional
import sys

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
    # Check if it has the solutions attribute (older API)
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

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 400, 500)

        # Create trackbars for each parameter
        self.trackbar_config = {
            'trail_intensity': (0, 100, 85),
            'trail_decay': (0, 100, 92),
            'particle_emission': (0, 20, 5),
            'glow_strength': (0, 100, 40),
            'scanline_intensity': (0, 50, 15),
            'chromatic_aberration': (0, 10, 2),
            'noise_level': (0, 20, 2),
            'brightness': (50, 200, 110),
            'contrast': (50, 200, 120),
            'vignette_strength': (0, 100, 30),
        }

        # Add skeleton opacity only if mediapipe available
        if has_skeleton:
            self.trackbar_config['skeleton_opacity'] = (0, 100, 70)

        for name, (min_val, max_val, default) in self.trackbar_config.items():
            cv2.createTrackbar(name, self.window_name, default, max_val, lambda x: None)

    def update_params(self):
        """Read trackbar values and update params dict"""
        try:
            self.params['trail_intensity'] = cv2.getTrackbarPos('trail_intensity', self.window_name) / 100.0
            self.params['trail_decay'] = cv2.getTrackbarPos('trail_decay', self.window_name) / 100.0
            self.params['particle_emission'] = cv2.getTrackbarPos('particle_emission', self.window_name)
            self.params['glow_strength'] = cv2.getTrackbarPos('glow_strength', self.window_name) / 100.0
            self.params['scanline_intensity'] = cv2.getTrackbarPos('scanline_intensity', self.window_name) / 100.0
            self.params['chromatic_aberration'] = cv2.getTrackbarPos('chromatic_aberration', self.window_name)
            self.params['noise_level'] = cv2.getTrackbarPos('noise_level', self.window_name) / 100.0
            self.params['brightness'] = cv2.getTrackbarPos('brightness', self.window_name) / 100.0
            self.params['contrast'] = cv2.getTrackbarPos('contrast', self.window_name) / 100.0
            self.params['vignette_strength'] = cv2.getTrackbarPos('vignette_strength', self.window_name) / 100.0

            if self.has_skeleton:
                self.params['skeleton_opacity'] = cv2.getTrackbarPos('skeleton_opacity', self.window_name) / 100.0
        except cv2.error:
            pass

    def show(self, fps, skeleton_active=False):
        """Display control panel with FPS"""
        info = np.zeros((500, 400, 3), dtype=np.uint8)
        info[:] = (30, 30, 30)

        # Title
        cv2.putText(info, "SCINTIGRAPHY CONTROLS", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

        # FPS
        fps_color = (0, 255, 100) if fps > 30 else (0, 100, 255)
        cv2.putText(info, f"FPS: {fps:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1)

        # Skeleton status
        if self.has_skeleton:
            status = "ACTIVE" if skeleton_active else "No body detected"
            status_color = (0, 255, 200) if skeleton_active else (100, 100, 100)
            cv2.putText(info, f"Skeleton: {status}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        else:
            cv2.putText(info, "Skeleton: DISABLED", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        # Instructions
        cv2.putText(info, "Use sliders above to adjust", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(info, "Press ESC to quit", (20, 475),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        cv2.imshow(self.window_name, info)


class ScintigraphyTracker:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Check pygame
        if not PYGAME_AVAILABLE:
            raise RuntimeError("pygame is required")

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Scintigraphy Blob Tracker")
        self.clock = pygame.time.Clock()

        # Webcam
        print("📹 Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Could not open webcam")

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
                    model_complexity=0,  # 0=lite (fastest)
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.landmark_style = mp_drawing.DrawingSpec(
                    color=(0, 255, 255), thickness=2, circle_radius=3
                )
                self.connection_style = mp_drawing.DrawingSpec(
                    color=(0, 200, 255), thickness=2
                )
                self.use_skeleton = True
                print("✓ Skeleton tracking initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize skeleton tracking: {e}")
                self.use_skeleton = False

        # Background subtractor
        print("📊 Using MOG2 background subtraction")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # Trail buffer
        self.trail_buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Particle system
        self.particles: List[Particle] = []
        self.max_particles = 500
        self.prev_centroid = None

        # Colormap LUT
        self.colormap_lut = self._create_scintigraphy_lut()

        # Parameters
        self.params = {
            'trail_intensity': 0.85,
            'trail_decay': 0.92,
            'particle_emission': 5,
            'glow_strength': 0.4,
            'scanline_intensity': 0.15,
            'chromatic_aberration': 2,
            'noise_level': 0.02,
            'brightness': 1.1,
            'contrast': 1.2,
            'vignette_strength': 0.3,
            'skeleton_opacity': 0.7,
        }

        # Control window
        self.control_window = ControlWindow(self.params, has_skeleton=self.use_skeleton)

        # FPS tracking
        self.fps_history = deque(maxlen=30)

        # Static textures
        self._create_static_textures()

        # Skeleton detection state
        self.skeleton_detected = False

    def _create_scintigraphy_lut(self):
        """Create colormap lookup table"""
        # Create 3 separate LUTs for B, G, R channels
        lut_b = np.zeros(256, dtype=np.uint8)
        lut_g = np.zeros(256, dtype=np.uint8)
        lut_r = np.zeros(256, dtype=np.uint8)

        for i in range(256):
            t = i / 255.0

            if t < 0.2:
                r, g, b = 0, 0, int(t * 5 * 180)
            elif t < 0.4:
                tt = (t - 0.2) / 0.2
                r, g, b = 0, int(tt * 200), 180 + int(tt * 75)
            elif t < 0.6:
                tt = (t - 0.4) / 0.2
                r, g, b = 0, 200 + int(tt * 55), int(255 * (1 - tt))
            elif t < 0.8:
                tt = (t - 0.6) / 0.2
                r, g, b = int(tt * 255), 255, 0
            else:
                tt = (t - 0.8) / 0.2
                r, g, b = 255, int(255 * (1 - tt * 0.7)), int(tt * 100)

            lut_b[i] = b
            lut_g[i] = g
            lut_r[i] = r

        return (lut_b, lut_g, lut_r)

    def _create_static_textures(self):
        """Pre-compute static textures"""
        self.scanlines = np.ones((self.height, self.width), dtype=np.float32)
        self.scanlines[1::2, :] = 0.85

        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        self.vignette = (1.0 - np.clip(dist * 0.6, 0, 0.6)).astype(np.float32)

    def detect_pose(self, frame):
        """Detect pose using MediaPipe"""
        if not self.use_skeleton or self.pose is None:
            return None

        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            self.skeleton_detected = results.pose_landmarks is not None
            return results
        except Exception as e:
            print(f"Pose detection error: {e}")
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

            # Glow effect
            skeleton_glow = cv2.GaussianBlur(skeleton_layer, (9, 9), 0)
            skeleton_layer = cv2.addWeighted(skeleton_layer, 1.0, skeleton_glow, 0.5, 0)

            # Blend
            opacity = self.params['skeleton_opacity']
            mask = cv2.cvtColor(skeleton_layer, cv2.COLOR_BGR2GRAY) > 0
            image[mask] = cv2.addWeighted(
                image, 1 - opacity,
                skeleton_layer, opacity, 0
            )[mask]
        except Exception as e:
            print(f"Skeleton draw error: {e}")

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
        # Apply colormap using separate LUTs for each channel
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

        return self.trail_buffer.copy()

    def emit_particles(self, contour, centroid):
        """Emit particles"""
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
                size=np.random.uniform(2, 4)
            )

            self.particles.append(particle)

        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

        self.prev_centroid = centroid

    def update_particles(self, dt):
        """Update particles"""
        alive_particles = []

        for particle in self.particles:
            particle.x += particle.vx * dt * 60
            particle.y += particle.vy * dt * 60
            particle.vx *= 0.96
            particle.vy *= 0.96
            particle.vy += 0.03
            particle.life -= dt * 2.0

            if particle.life > 0:
                alive_particles.append(particle)

        self.particles = alive_particles

    def apply_effects_fast(self, image):
        """Apply visual effects"""
        result = image.astype(np.float32)

        result = result * self.params['contrast'] * self.params['brightness']

        vignette_strength = self.params['vignette_strength']
        vignette_factor = 1.0 - vignette_strength * (1.0 - self.vignette[:, :, np.newaxis])
        result = result * vignette_factor

        scanline_factor = 1.0 - self.params['scanline_intensity'] * (1.0 - self.scanlines[:, :, np.newaxis])
        result = result * scanline_factor

        if self.params['noise_level'] > 0:
            noise = np.random.randint(-10, 10, (self.height, self.width, 1), dtype=np.int16)
            noise = noise * self.params['noise_level']
            result = result + noise

        result = np.clip(result, 0, 255).astype(np.uint8)

        offset = int(self.params['chromatic_aberration'])
        if offset > 0:
            b, g, r = cv2.split(result)
            r = np.roll(r, offset, axis=1)
            b = np.roll(b, -offset, axis=1)
            result = cv2.merge([b, g, r])

        if self.params['glow_strength'] > 0:
            glow = cv2.GaussianBlur(result, (7, 7), 0)
            result = cv2.addWeighted(result, 1.0, glow, self.params['glow_strength'], 0)

        return result

    def render_particles_fast(self, surface):
        """Render particles"""
        for particle in self.particles:
            if 0 <= particle.x < self.width and 0 <= particle.y < self.height:
                life = particle.life
                intensity = particle.intensity * life

                if life > 0.5:
                    color = (
                        int(255 * intensity),
                        int(180 * intensity),
                        int(50 * intensity)
                    )
                else:
                    color = (
                        int(100 * intensity),
                        int(200 * intensity),
                        int(255 * intensity)
                    )

                size = int(particle.size * life)
                if size > 0:
                    pygame.draw.circle(
                        surface,
                        color,
                        (int(particle.x), int(particle.y)),
                        size
                    )

    def handle_input(self):
        """Handle input"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False
        return True

    def cleanup(self):
        """Cleanup resources"""
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
        if self.use_skeleton:
            print("   + MediaPipe Skeleton Tracking")
        print("   + Separate Control Window")
        print("="*60)
        print("\n⌨️  Press ESC to quit")
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
                    print("❌ Failed to capture frame")
                    break

                frame = cv2.flip(frame, 1)

                # Pose detection (if available)
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

                # Skeleton overlay
                if self.use_skeleton:
                    final_image = self.draw_skeleton(final_image, pose_results)

                # Convert to Pygame
                final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
                final_image_rgb = np.rot90(final_image_rgb)
                final_image_rgb = np.flipud(final_image_rgb)
                pygame_surface = pygame.surfarray.make_surface(final_image_rgb)

                self.screen.blit(pygame_surface, (0, 0))
                self.render_particles_fast(self.screen)
                pygame.display.flip()

                # Control window
                avg_fps = np.mean(self.fps_history) if self.fps_history else 0
                self.control_window.show(avg_fps, self.skeleton_detected)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    running = False

                self.clock.tick(60)

        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        finally:
            self.cleanup()
            print("\n✨ Scintigraphy Tracker closed.")


def main():
    """Entry point"""
    print("\n" + "="*60)
    print("🔬 SCINTIGRAPHY BLOB TRACKER")
    print("   Medical Scifi Real-time Performance Tool")
    print("="*60 + "\n")

    try:
        tracker = ScintigraphyTracker(width=1280, height=720)
        tracker.run()
    except Exception as e:
        print(f"❌ Failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
