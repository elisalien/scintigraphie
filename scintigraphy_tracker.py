"""
Scintigraphy-Inspired Blob Tracker with CRT Effects
Real-time webcam tracking with medical/scifi aesthetic
"""

import cv2
import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import List, Tuple
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
        
        # Background removal
        self.bg_session = None
        self.use_rembg = REMBG_AVAILABLE
        
        if self.use_rembg:
            print("🚀 Initializing NVIDIA-accelerated background removal...")
            try:
                self.bg_session = new_session("u2net")  # Fast model
            except Exception as e:
                print(f"⚠️  Could not initialize GPU session: {e}")
                self.use_rembg = False
        
        # Fallback: background subtractor (always created as fallback)
        if not self.use_rembg:
            print("📊 Using fallback background subtraction (MOG2)")
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Heatmap accumulation (for temporal trails)
        self.heatmap = np.zeros((height, width), dtype=np.float32)
        self.heatmap_decay = 0.95  # Decay rate for trails
        
        # Particle system for radioactive effect
        self.particles: List[Particle] = []
        self.max_particles = 2000
        
        # Blob tracking
        self.blob_history = deque(maxlen=30)  # 30 frame history
        self.prev_centroid = None
        
        # CRT/Scifi parameters (adjustable)
        self.params = {
            'heatmap_intensity': 1.2,
            'trail_decay': 0.92,
            'particle_emission': 8,  # particles per frame
            'glow_radius': 9,
            'scanline_intensity': 0.3,
            'chromatic_aberration': 2.0,
            'phosphor_persistence': 0.6,
            'noise_level': 0.05,
            'gamma': 1.2,
            'contrast': 1.3,
        }
        
        # UI state
        self.show_ui = True
        self.selected_param = 0
        self.param_names = list(self.params.keys())
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        
        # CRT effect textures
        self.create_crt_textures()
        
    def create_crt_textures(self):
        """Create textures for CRT effects"""
        # Scanlines
        self.scanlines = np.zeros((self.height, self.width), dtype=np.float32)
        for y in range(self.height):
            intensity = 1.0 - (y % 3) * 0.2  # Every 3rd line darker
            self.scanlines[y, :] = intensity
            
        # Vignette (CRT screen curvature simulation)
        cy, cx = self.height // 2, self.width // 2
        y, x = np.ogrid[:self.height, :self.width]
        dist = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2)
        self.vignette = 1.0 - np.clip(dist * 0.5, 0, 0.5)
        
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
    
    def update_heatmap(self, mask):
        """Update heatmap with new blob data"""
        # Decay existing heatmap
        self.heatmap *= self.params['trail_decay']
        
        # Add new intensity from mask
        normalized_mask = mask.astype(np.float32) / 255.0
        self.heatmap += normalized_mask * self.params['heatmap_intensity']
        
        # Clip to prevent overflow
        self.heatmap = np.clip(self.heatmap, 0, 10.0)
    
    def emit_particles(self, contour, centroid):
        """Emit radioactive particles from blob edges"""
        if contour is None or centroid is None:
            return
        
        # Calculate velocity from blob history
        velocity = (0, 0)
        if self.prev_centroid and centroid:
            velocity = (
                centroid[0] - self.prev_centroid[0],
                centroid[1] - self.prev_centroid[1]
            )
        
        # Emit particles along contour edges
        num_particles = int(self.params['particle_emission'])
        
        for _ in range(num_particles):
            # Random point on contour
            idx = np.random.randint(0, len(contour))
            point = contour[idx][0]
            
            # Calculate normal direction (outward from blob)
            if centroid:
                dx = point[0] - centroid[0]
                dy = point[1] - centroid[1]
                dist = np.sqrt(dx**2 + dy**2) + 0.001
                dx /= dist
                dy /= dist
            else:
                dx, dy = np.random.randn(2)
            
            # Create particle with velocity
            particle = Particle(
                x=float(point[0]),
                y=float(point[1]),
                vx=dx * np.random.uniform(1, 3) + velocity[0] * 0.3,
                vy=dy * np.random.uniform(1, 3) + velocity[1] * 0.3,
                life=1.0,
                intensity=np.random.uniform(0.5, 1.0),
                size=np.random.uniform(1, 3)
            )
            
            self.particles.append(particle)
        
        # Limit particle count
        if len(self.particles) > self.max_particles:
            self.particles = self.particles[-self.max_particles:]
        
        # Update previous centroid
        self.prev_centroid = centroid
    
    def update_particles(self, dt):
        """Update particle physics"""
        for particle in self.particles[:]:
            # Update position
            particle.x += particle.vx * dt * 60
            particle.y += particle.vy * dt * 60
            
            # Apply drag
            particle.vx *= 0.98
            particle.vy *= 0.98
            
            # Gravity effect (slight)
            particle.vy += 0.05
            
            # Decay life
            particle.life -= dt * 1.5
            
            # Remove dead particles
            if particle.life <= 0:
                self.particles.remove(particle)
    
    def apply_colormap(self, heatmap):
        """Apply medical scintigraphy colormap to heatmap"""
        # Normalize heatmap
        heatmap_norm = np.clip(heatmap / 10.0, 0, 1.0)

        # Apply gamma correction
        heatmap_norm = np.power(heatmap_norm, 1.0 / self.params['gamma'])

        # Medical scintigraphy colormap: Black -> Blue -> Cyan -> Green -> Yellow -> Red -> White
        h, w = heatmap_norm.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Piecewise linear colormap matching nuclear medicine imaging
        t = heatmap_norm

        # Red channel: low until 0.5, then rises to 1.0
        r = np.where(t < 0.4, 0,
            np.where(t < 0.75, (t - 0.4) / 0.35,
            1.0))

        # Green channel: rises from 0.2 to 0.5, then stays high until 0.8
        g = np.where(t < 0.15, 0,
            np.where(t < 0.4, (t - 0.15) / 0.25 * 0.8,
            np.where(t < 0.75, 0.8 + (t - 0.4) * 0.5,
            1.0 - (t - 0.75) * 0.5)))

        # Blue channel: high at start, fades out in middle, rises again at hot spots
        b = np.where(t < 0.1, t * 3,
            np.where(t < 0.4, 0.3 + (t - 0.1) * 2,
            np.where(t < 0.6, 0.9 - (t - 0.4) * 3,
            np.where(t < 0.85, 0.3 - (t - 0.6) * 1.2,
            (t - 0.85) * 5))))

        colored[:, :, 0] = np.clip(r * 255, 0, 255).astype(np.uint8)  # Red (BGR)
        colored[:, :, 1] = np.clip(g * 255, 0, 255).astype(np.uint8)  # Green
        colored[:, :, 2] = np.clip(b * 255, 0, 255).astype(np.uint8)  # Blue

        return colored
    
    def apply_glow(self, image):
        """Apply phosphor-like glow effect"""
        glow_radius = int(self.params['glow_radius'])
        
        # Gaussian blur for glow
        glowed = cv2.GaussianBlur(image, (glow_radius * 2 + 1, glow_radius * 2 + 1), 0)
        
        # Blend original with glow
        persistence = self.params['phosphor_persistence']
        result = cv2.addWeighted(image, 1.0, glowed, persistence, 0)
        
        return result
    
    def apply_chromatic_aberration(self, image):
        """Apply chromatic aberration (color fringing) for CRT effect"""
        offset = int(self.params['chromatic_aberration'])
        
        if offset == 0:
            return image
        
        # Split channels
        b, g, r = cv2.split(image)
        
        # Shift red channel right
        r_shifted = np.roll(r, offset, axis=1)
        
        # Shift blue channel left
        b_shifted = np.roll(b, -offset, axis=1)
        
        # Merge
        result = cv2.merge([b_shifted, g, r_shifted])
        
        return result
    
    def apply_crt_effects(self, image):
        """Apply all CRT effects"""
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Apply scanlines
        for c in range(3):
            img_float[:, :, c] *= (1.0 - self.params['scanline_intensity'] * (1.0 - self.scanlines))
        
        # Apply vignette
        for c in range(3):
            img_float[:, :, c] *= self.vignette
        
        # Add noise (analog interference)
        noise = np.random.randn(self.height, self.width) * self.params['noise_level']
        for c in range(3):
            img_float[:, :, c] += noise
        
        # Contrast adjustment
        img_float = (img_float - 0.5) * self.params['contrast'] + 0.5
        
        # Clip and convert back
        img_float = np.clip(img_float, 0, 1.0)
        result = (img_float * 255).astype(np.uint8)
        
        return result
    
    def render_particles(self, surface):
        """Render radioactive particles with glow"""
        for particle in self.particles:
            if 0 <= particle.x < self.width and 0 <= particle.y < self.height:
                # Color based on life (hot to cold)
                life_ratio = particle.life
                
                # Hot colors when fresh
                if life_ratio > 0.7:
                    color = (
                        int(255 * particle.intensity),
                        int(200 * particle.intensity * life_ratio),
                        int(50 * particle.intensity * life_ratio)
                    )
                else:
                    # Cool down to cyan/blue
                    color = (
                        int(50 * particle.intensity * life_ratio),
                        int(180 * particle.intensity * life_ratio),
                        int(255 * particle.intensity * life_ratio)
                    )
                
                # Draw particle with glow
                alpha = int(255 * life_ratio * particle.intensity)
                size = int(particle.size * (1 + (1 - life_ratio) * 0.5))
                
                # Glow layers
                for i in range(3):
                    glow_size = size + i * 2
                    glow_alpha = alpha // (i + 2)
                    glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surf, (*color, glow_alpha), (glow_size, glow_size), glow_size)
                    surface.blit(glow_surf, (int(particle.x - glow_size), int(particle.y - glow_size)))
    
    def render_ui(self, surface):
        """Render parameter adjustment UI"""
        if not self.show_ui:
            return
        
        # Semi-transparent background
        ui_rect = pygame.Surface((350, 400), pygame.SRCALPHA)
        ui_rect.fill((0, 20, 40, 200))
        surface.blit(ui_rect, (10, 10))
        
        # Title
        title = self.font.render("SCINTIGRAPHY TRACKER", True, (0, 255, 200))
        surface.blit(title, (20, 20))
        
        # FPS
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        fps_text = self.font_small.render(f"FPS: {avg_fps:.1f}", True, (0, 255, 100))
        surface.blit(fps_text, (20, 50))
        
        # Parameters
        y_offset = 90
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
            y_offset += 25
        
        # Instructions
        instructions = [
            "",
            "Controls:",
            "UP/DOWN: Select parameter",
            "LEFT/RIGHT: Adjust value",
            "H: Toggle UI",
            "R: Reset parameters",
            "ESC: Quit"
        ]
        
        y_offset += 20
        for instruction in instructions:
            text = self.font_small.render(instruction, True, (100, 150, 200))
            surface.blit(text, (20, y_offset))
            y_offset += 20
    
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
                    self.params[param_name] = max(0.0, self.params[param_name] - 0.1)
                
                elif event.key == K_RIGHT:
                    param_name = self.param_names[self.selected_param]
                    self.params[param_name] = min(10.0, self.params[param_name] + 0.1)
                
                elif event.key == K_r:
                    # Reset to defaults
                    self.params = {
                        'heatmap_intensity': 1.2,
                        'trail_decay': 0.92,
                        'particle_emission': 8,
                        'glow_radius': 9,
                        'scanline_intensity': 0.3,
                        'chromatic_aberration': 2.0,
                        'phosphor_persistence': 0.6,
                        'noise_level': 0.05,
                        'gamma': 1.2,
                        'contrast': 1.3,
                    }
        
        return True
    
    def run(self):
        """Main loop"""
        print("\n" + "="*60)
        print("🔬 SCINTIGRAPHY BLOB TRACKER INITIALIZED")
        print("="*60)
        print("\n📹 Starting webcam capture...")
        print("🎨 Medical scifi rendering active")
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
            
            # Update heatmap
            self.update_heatmap(mask)
            
            # Emit and update particles
            if contour is not None:
                self.emit_particles(contour, centroid)
            self.update_particles(dt)
            
            # Apply colormap to heatmap
            colored_heatmap = self.apply_colormap(self.heatmap)
            
            # Apply glow effect
            glowed = self.apply_glow(colored_heatmap)
            
            # Apply chromatic aberration
            aberrated = self.apply_chromatic_aberration(glowed)
            
            # Apply CRT effects
            final_image = self.apply_crt_effects(aberrated)
            
            # Convert to Pygame surface
            final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
            final_image = np.rot90(final_image)
            final_image = np.flipud(final_image)
            pygame_surface = pygame.surfarray.make_surface(final_image)
            
            # Blit to screen
            self.screen.blit(pygame_surface, (0, 0))
            
            # Render particles on top
            self.render_particles(self.screen)
            
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
