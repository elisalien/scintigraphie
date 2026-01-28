"""
Scintigraphy-Inspired Blob Tracker with CRT Effects
Real-time webcam tracking with medical/scifi aesthetic
OPTIMIZED VERSION - 60+ FPS target
"""

import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List
import pygame
from pygame.locals import *

# Try to import rembg for GPU-accelerated background removal
try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️  rembg not installed. Using basic background subtraction.")


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


class ScintigraphyTracker:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height

        # Initialize Pygame for rendering
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Scintigraphy Blob Tracker - Medical Scifi")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)

        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Background removal - MOG2 by default (fast), rembg optional (requires CUDA)
        self.bg_session = None
        self.use_rembg = False  # Disabled by default - MOG2 is much faster

        # To enable rembg (slower, requires CUDA), set use_rembg = True above
        if self.use_rembg and REMBG_AVAILABLE:
            print("🚀 Initializing NVIDIA-accelerated background removal...")
            try:
                self.bg_session = new_session("u2net")
            except Exception as e:
                print(f"⚠️  Could not initialize GPU session: {e}")
                self.use_rembg = False

        # Fallback: background subtractor (always created as fallback)
        if not self.use_rembg:
            print("📊 Using fallback background subtraction (MOG2)")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

        # Trail buffer - simple frame blending for trails (FAST)
        self.trail_buffer = np.zeros((height, width, 3), dtype=np.uint8)

        # Particle system for radioactive effect
        self.particles: List[Particle] = []
        self.max_particles = 500  # Reduced for performance

        # Blob tracking
        self.prev_centroid = None

        # Pre-create colormap LUT (FAST lookup instead of np.where)
        self.colormap_lut = self._create_scintigraphy_lut()

        # CRT/Scifi parameters (adjustable)
        self.params = {
            'trail_intensity': 0.85,      # Trail persistence (0-1)
            'trail_decay': 0.92,          # How fast trails fade
            'particle_emission': 5,       # particles per frame (reduced)
            'glow_strength': 0.4,         # Glow blend amount
            'scanline_intensity': 0.15,   # CRT scanlines
            'chromatic_aberration': 2.0,  # Color fringing
            'noise_level': 0.02,          # Analog noise
            'brightness': 1.1,            # Overall brightness
            'contrast': 1.2,              # Contrast
            'vignette_strength': 0.3,     # Edge darkening
        }

        # UI state
        self.show_ui = True
        self.selected_param = 0
        self.param_names = list(self.params.keys())

        # FPS tracking
        self.fps_history = deque(maxlen=30)

        # Pre-compute static textures
        self._create_static_textures()

    def _create_scintigraphy_lut(self):
        """Create a 256-entry lookup table for fast colormap application"""
        lut = np.zeros((256, 1, 3), dtype=np.uint8)

        for i in range(256):
            t = i / 255.0

            # Medical scintigraphy colors: Black -> Blue -> Cyan -> Green -> Yellow -> Red
            if t < 0.2:
                # Black to deep blue
                r, g, b = 0, 0, int(t * 5 * 180)
            elif t < 0.4:
                # Deep blue to cyan
                tt = (t - 0.2) / 0.2
                r, g, b = 0, int(tt * 200), 180 + int(tt * 75)
            elif t < 0.6:
                # Cyan to green
                tt = (t - 0.4) / 0.2
                r, g, b = 0, 200 + int(tt * 55), int(255 * (1 - tt))
            elif t < 0.8:
                # Green to yellow
                tt = (t - 0.6) / 0.2
                r, g, b = int(tt * 255), 255, 0
            else:
                # Yellow to red/white
                tt = (t - 0.8) / 0.2
                r, g, b = 255, int(255 * (1 - tt * 0.7)), int(tt * 100)

            lut[i, 0] = [b, g, r]  # BGR format

        return lut

    def _create_static_textures(self):
        """Pre-compute static effect textures"""
        # Scanlines texture (every 2nd line darker)
        self.scanlines = np.ones((self.height, self.width), dtype=np.float32)
        self.scanlines[1::2, :] = 0.85

        # Vignette (edge darkening)
        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        self.vignette = (1.0 - np.clip(dist * 0.6, 0, 0.6)).astype(np.float32)

        # Convert to uint8 for fast blending
        self.vignette_u8 = (self.vignette * 255).astype(np.uint8)

    def remove_background(self, frame):
        """Remove background using NVIDIA or fallback method"""
        if self.use_rembg and self.bg_session:
            try:
                from PIL import Image
                import io

                # Encode frame to JPEG (much faster than PNG)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buffer.tobytes()

                # Remove background (pass bytes, get bytes back)
                output_bytes = remove(img_bytes, session=self.bg_session,
                                      force_return_bytes=True)

                # Decode output bytes back to image
                output_img = Image.open(io.BytesIO(output_bytes))
                output_np = np.array(output_img)

                # Extract alpha channel as mask
                if len(output_np.shape) == 3 and output_np.shape[2] == 4:
                    mask = output_np[:, :, 3]
                elif len(output_np.shape) == 3:
                    mask = cv2.cvtColor(output_np, cv2.COLOR_RGB2GRAY)
                else:
                    mask = output_np

                return mask
            except Exception as e:
                print(f"Background removal error: {e}")
                return self.bg_subtractor.apply(frame)
        else:
            # Fallback: basic background subtraction
            return self.bg_subtractor.apply(frame)

    def detect_blob(self, mask):
        """Detect main blob (person) in mask"""
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Get largest contour (main person)
        largest_contour = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest_contour) < 5000:  # Minimum area threshold
            return None, None

        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroid = (cx, cy)
        else:
            centroid = None

        return largest_contour, centroid

    def update_trail(self, mask):
        """Update trail buffer with simple blend (FAST)"""
        # Convert mask to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Apply colormap using LUT (FAST)
        colored = cv2.LUT(mask, self.colormap_lut)

        # Blend with trail buffer for persistence
        decay = self.params['trail_decay']
        intensity = self.params['trail_intensity']

        # Fast trail: decay old + add new
        self.trail_buffer = cv2.addWeighted(
            self.trail_buffer, decay,
            colored, intensity * (1 - decay),
            0
        )

        return self.trail_buffer.copy()

    def emit_particles(self, contour, centroid):
        """Emit radioactive particles from blob edges"""
        if contour is None or centroid is None:
            return

        # Calculate velocity from blob movement
        velocity = (0, 0)
        if self.prev_centroid and centroid:
            velocity = (
                centroid[0] - self.prev_centroid[0],
                centroid[1] - self.prev_centroid[1]
            )

        # Emit particles along contour edges
        num_particles = int(self.params['particle_emission'])
        contour_len = len(contour)

        if contour_len == 0:
            return

        for _ in range(num_particles):
            # Random point on contour
            idx = np.random.randint(0, contour_len)
            point = contour[idx][0]

            # Direction outward from blob
            dx = point[0] - centroid[0]
            dy = point[1] - centroid[1]
            dist = np.sqrt(dx**2 + dy**2) + 0.001
            dx /= dist
            dy /= dist

            # Create particle
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

        # Limit particle count
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]

        # Update previous centroid
        self.prev_centroid = centroid

    def update_particles(self, dt):
        """Update particle physics (optimized)"""
        # Process particles in batch where possible
        alive_particles = []

        for particle in self.particles:
            # Update position
            particle.x += particle.vx * dt * 60
            particle.y += particle.vy * dt * 60

            # Apply drag
            particle.vx *= 0.96
            particle.vy *= 0.96

            # Gravity
            particle.vy += 0.03

            # Decay life
            particle.life -= dt * 2.0

            # Keep alive particles
            if particle.life > 0:
                alive_particles.append(particle)

        self.particles = alive_particles

    def apply_effects_fast(self, image):
        """Apply all visual effects in one optimized pass"""
        # Start with the image
        result = image.astype(np.float32)

        # Apply brightness/contrast
        result = result * self.params['contrast'] * self.params['brightness']

        # Apply vignette (pre-computed)
        vignette_strength = self.params['vignette_strength']
        vignette_factor = 1.0 - vignette_strength * (1.0 - self.vignette[:, :, np.newaxis])
        result = result * vignette_factor

        # Apply scanlines
        scanline_factor = 1.0 - self.params['scanline_intensity'] * (1.0 - self.scanlines[:, :, np.newaxis])
        result = result * scanline_factor

        # Add subtle noise
        if self.params['noise_level'] > 0:
            noise = np.random.randint(-10, 10, (self.height, self.width, 1), dtype=np.int16)
            noise = noise * self.params['noise_level']
            result = result + noise

        # Clip and convert
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Chromatic aberration (simple shift)
        offset = int(self.params['chromatic_aberration'])
        if offset > 0:
            b, g, r = cv2.split(result)
            r = np.roll(r, offset, axis=1)
            b = np.roll(b, -offset, axis=1)
            result = cv2.merge([b, g, r])

        # Fast glow using small blur
        if self.params['glow_strength'] > 0:
            glow = cv2.GaussianBlur(result, (7, 7), 0)
            result = cv2.addWeighted(result, 1.0, glow, self.params['glow_strength'], 0)

        return result

    def render_particles_fast(self, surface):
        """Render particles with minimal overhead"""
        for particle in self.particles:
            if 0 <= particle.x < self.width and 0 <= particle.y < self.height:
                # Simple color based on life
                life = particle.life
                intensity = particle.intensity * life

                # Hot to cold color transition
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

                # Draw simple circle (no alpha blending for speed)
                size = int(particle.size * life)
                if size > 0:
                    pygame.draw.circle(
                        surface,
                        color,
                        (int(particle.x), int(particle.y)),
                        size
                    )

    def render_ui(self, surface):
        """Render parameter adjustment UI"""
        if not self.show_ui:
            return

        # Semi-transparent background
        ui_rect = pygame.Surface((320, 380), pygame.SRCALPHA)
        ui_rect.fill((0, 20, 40, 180))
        surface.blit(ui_rect, (10, 10))

        # Title
        title = self.font.render("SCINTIGRAPHY TRACKER", True, (0, 255, 200))
        surface.blit(title, (20, 20))

        # FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        fps_color = (0, 255, 100) if avg_fps > 30 else (255, 100, 0)
        fps_text = self.font_small.render(f"FPS: {avg_fps:.1f}", True, fps_color)
        surface.blit(fps_text, (20, 50))

        # Parameters
        y_offset = 80
        for i, (name, value) in enumerate(self.params.items()):
            # Highlight selected parameter
            if i == self.selected_param:
                color = (255, 255, 0)
                prefix = "> "
            else:
                color = (150, 200, 255)
                prefix = "  "

            # Format parameter name
            display_name = name.replace('_', ' ').title()
            text = self.font_small.render(f"{prefix}{display_name}: {value:.2f}", True, color)
            surface.blit(text, (20, y_offset))
            y_offset += 22

        # Instructions
        instructions = [
            "",
            "UP/DOWN: Select | LEFT/RIGHT: Adjust",
            "H: Toggle UI | R: Reset | ESC: Quit"
        ]

        y_offset += 10
        for instruction in instructions:
            text = self.font_small.render(instruction, True, (100, 150, 200))
            surface.blit(text, (20, y_offset))
            y_offset += 18

    def handle_input(self):
        """Handle keyboard input for parameter adjustment"""
        for event in pygame.event.get():
            if event.type == QUIT:
                return False

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return False

                elif event.key == K_h:
                    self.show_ui = not self.show_ui

                elif event.key == K_UP:
                    self.selected_param = (self.selected_param - 1) % len(self.param_names)

                elif event.key == K_DOWN:
                    self.selected_param = (self.selected_param + 1) % len(self.param_names)

                elif event.key == K_LEFT:
                    param_name = self.param_names[self.selected_param]
                    self.params[param_name] = max(0.0, self.params[param_name] - 0.05)

                elif event.key == K_RIGHT:
                    param_name = self.param_names[self.selected_param]
                    self.params[param_name] = min(2.0, self.params[param_name] + 0.05)

                elif event.key == K_r:
                    # Reset to defaults
                    self.params = {
                        'trail_intensity': 0.85,
                        'trail_decay': 0.92,
                        'particle_emission': 5,
                        'glow_strength': 0.4,
                        'scanline_intensity': 0.15,
                        'chromatic_aberration': 2.0,
                        'noise_level': 0.02,
                        'brightness': 1.1,
                        'contrast': 1.2,
                        'vignette_strength': 0.3,
                    }

        return True

    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("🔬 SCINTIGRAPHY BLOB TRACKER INITIALIZED")
        print("="*60)
        print("\n📹 Starting webcam capture...")
        print("🎨 Medical scifi rendering active (OPTIMIZED)")
        print("\n⌨️  Controls:")
        print("   UP/DOWN: Select parameter")
        print("   LEFT/RIGHT: Adjust value")
        print("   H: Toggle UI")
        print("   R: Reset parameters")
        print("   ESC: Quit")
        print("\n" + "="*60 + "\n")

        running = True
        last_time = time.time()

        while running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            # Track FPS
            if dt > 0:
                self.fps_history.append(1.0 / dt)

            # Handle input
            running = self.handle_input()
            if not running:
                break

            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break

            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Remove background
            mask = self.remove_background(frame)

            # Detect blob
            contour, centroid = self.detect_blob(mask)

            # Update trail with colormap
            colored_trail = self.update_trail(mask)

            # Emit and update particles
            if contour is not None:
                self.emit_particles(contour, centroid)
            self.update_particles(dt)

            # Apply all effects (optimized single pass)
            final_image = self.apply_effects_fast(colored_trail)

            # Convert to Pygame surface
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            final_image = np.rot90(final_image)
            final_image = np.flipud(final_image)
            pygame_surface = pygame.surfarray.make_surface(final_image)

            # Blit to screen
            self.screen.blit(pygame_surface, (0, 0))

            # Render particles on top
            self.render_particles_fast(self.screen)

            # Render UI
            self.render_ui(self.screen)

            # Update display
            pygame.display.flip()

            # Target 60 FPS
            self.clock.tick(60)

        # Cleanup
        self.cap.release()
        pygame.quit()
        print("\n✨ Scintigraphy Tracker closed. À bientôt!")


def main():
    """Entry point"""
    print("\n" + "="*60)
    print("🔬 SCINTIGRAPHY BLOB TRACKER")
    print("   Medical Scifi Real-time Performance Tool")
    print("="*60 + "\n")

    tracker = ScintigraphyTracker(width=1280, height=720)
    tracker.run()


if __name__ == "__main__":
    main()
