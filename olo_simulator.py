#!/usr/bin/env python3
"""
OLO Color Simulator - Low-Level Pixel Technical Implementation
===============================================================

Attempts to simulate the novel color "olo" from Fong et al. (2025)
"Novel color via stimulation of individual photoreceptors at population scale"

True olo = Pure M-cone stimulation (L=0, M=1, S=0)
This is IMPOSSIBLE with spectral light on any display.

This script uses perceptual techniques:
1. Chromatic adaptation (bleach L-cones with red)
2. Optimal M-dominant RGB calculation
3. Temporal dithering/flicker fusion
4. Subpixel pattern exploitation
5. Opponent color surrounds
6. Von Kries adaptation modeling
"""

import numpy as np
import time
import sys
from dataclasses import dataclass
from typing import Tuple, List, Optional
import colorsys

# Try to import display libraries
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("pygame not found - install with: pip install pygame")

# ============================================================================
# COLOR SCIENCE CORE
# ============================================================================

@dataclass
class ConeResponse:
    """Container for L, M, S cone responses."""
    L: float
    M: float
    S: float
    
    @property
    def lms_chromaticity(self) -> Tuple[float, float, float]:
        """Return (l, m, s) chromaticity coordinates."""
        total = self.L + self.M + self.S
        if total < 1e-10:
            return (0.333, 0.333, 0.334)
        return (self.L/total, self.M/total, self.S/total)
    
    @property
    def m_dominance(self) -> float:
        """M / (L + S) ratio - key metric for olo approximation."""
        denominator = self.L + self.S
        if denominator < 1e-10:
            return float('inf')
        return self.M / denominator


class StockmanSharpeFundamentals:
    """
    Stockman-Sharpe (2000) cone fundamentals implementation.
    These are the standard for colorimetric calculations.
    """
    
    # Wavelength range
    LAMBDA_MIN = 390
    LAMBDA_MAX = 700
    
    # Pre-computed coefficients for log-Gaussian approximation
    # Parameters fitted to Stockman-Sharpe 2° fundamentals
    
    # L-cone parameters
    L_PARAMS = {'lambda_max': 566.8, 'sigma': 0.0355, 'beta': 0.00114}
    # M-cone parameters  
    M_PARAMS = {'lambda_max': 541.2, 'sigma': 0.0325, 'beta': 0.00095}
    # S-cone parameters
    S_PARAMS = {'lambda_max': 441.4, 'sigma': 0.0290, 'beta': 0.00155}
    
    @classmethod
    def _log_gaussian_sensitivity(cls, wavelength: float, params: dict) -> float:
        """Calculate cone sensitivity using log-Gaussian model."""
        lam = wavelength
        lam_max = params['lambda_max']
        sigma = params['sigma']
        
        # Log-parabola approximation
        x = np.log10(lam / lam_max)
        sensitivity = np.exp(-0.5 * (x / sigma) ** 2)
        
        return max(0, sensitivity)
    
    @classmethod
    def get_cone_responses(cls, wavelength: float) -> ConeResponse:
        """Get L, M, S cone responses to a monochromatic wavelength."""
        L = cls._log_gaussian_sensitivity(wavelength, cls.L_PARAMS)
        M = cls._log_gaussian_sensitivity(wavelength, cls.M_PARAMS)
        S = cls._log_gaussian_sensitivity(wavelength, cls.S_PARAMS)
        return ConeResponse(L, M, S)


class DisplayColorScience:
    """
    Color science calculations for display RGB to cone responses.
    Uses sRGB primary spectra and proper gamma handling.
    """
    
    # sRGB to XYZ transformation matrix (D65 illuminant)
    sRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # XYZ to LMS (Hunt-Pointer-Estevez, D65-normalized)
    XYZ_TO_LMS = np.array([
        [ 0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340,  0.04641],
        [ 0.00000, 0.00000,  1.00000]
    ])
    
    # Combined transform
    sRGB_TO_LMS = XYZ_TO_LMS @ sRGB_TO_XYZ
    LMS_TO_sRGB = np.linalg.inv(sRGB_TO_LMS)
    
    @classmethod
    def gamma_expand(cls, c: float) -> float:
        """sRGB gamma expansion (inverse EOTF)."""
        if c <= 0.04045:
            return c / 12.92
        return ((c + 0.055) / 1.055) ** 2.4
    
    @classmethod
    def gamma_compress(cls, c: float) -> float:
        """sRGB gamma compression (EOTF)."""
        if c <= 0.0031308:
            return c * 12.92
        return 1.055 * (c ** (1/2.4)) - 0.055
    
    @classmethod
    def rgb_to_lms(cls, r: int, g: int, b: int) -> ConeResponse:
        """Convert sRGB (0-255) to LMS cone responses."""
        # Normalize and gamma expand
        rgb_normalized = np.array([r, g, b]) / 255.0
        rgb_linear = np.array([cls.gamma_expand(c) for c in rgb_normalized])
        
        # Transform to LMS
        lms = cls.sRGB_TO_LMS @ rgb_linear
        
        return ConeResponse(
            L=max(0, lms[0]),
            M=max(0, lms[1]),
            S=max(0, lms[2])
        )
    
    @classmethod
    def lms_to_rgb(cls, L: float, M: float, S: float) -> Tuple[int, int, int]:
        """
        Convert LMS to sRGB. 
        Note: Pure M (L=0, S=0) is OUT OF GAMUT!
        Returns clipped values and a flag.
        """
        lms = np.array([L, M, S])
        rgb_linear = cls.LMS_TO_sRGB @ lms
        
        # Gamma compress
        rgb_normalized = np.array([cls.gamma_compress(max(0, c)) for c in rgb_linear])
        
        # Scale and clip
        rgb = np.clip(rgb_normalized * 255, 0, 255).astype(int)
        
        return tuple(rgb)
    
    @classmethod
    def find_max_m_dominance_rgb(cls, resolution: int = 256) -> Tuple[Tuple[int, int, int], float]:
        """
        Exhaustively search for sRGB color with maximum M/(L+S) ratio.
        This is the closest we can get to olo on a display.
        """
        best_rgb = (0, 255, 200)
        best_ratio = 0
        
        print("Searching for optimal olo approximation RGB...")
        
        for r in range(0, 64, 2):  # Low red reduces L-cone response
            for g in range(160, 256, 1):  # High green for M-cone
                for b in range(120, 240, 1):  # Moderate blue
                    resp = cls.rgb_to_lms(r, g, b)
                    ratio = resp.m_dominance
                    
                    # Ensure visible luminance
                    luminance = 0.2126*r + 0.7152*g + 0.0722*b
                    if ratio > best_ratio and luminance > 100 and ratio < 100:
                        best_ratio = ratio
                        best_rgb = (r, g, b)
        
        return best_rgb, best_ratio


class VonKriesAdaptation:
    """
    Von Kries chromatic adaptation model.
    Models how staring at a color changes subsequent color perception.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset to fully adapted (neutral) state."""
        self.l_gain = 1.0
        self.m_gain = 1.0
        self.s_gain = 1.0
    
    def adapt(self, stimulus: ConeResponse, duration_seconds: float, 
              adaptation_rate: float = 0.1):
        """
        Simulate adaptation to a stimulus over time.
        
        The visual system reduces sensitivity to sustained stimulation.
        Rate follows an exponential decay model.
        """
        # Adaptation follows: gain = 1 / (1 + k * stimulus * time)
        k = adaptation_rate
        t = duration_seconds
        
        # Calculate new gains (sensitivity reduction)
        if stimulus.L > 0.01:
            self.l_gain = 1.0 / (1.0 + k * stimulus.L * t)
        if stimulus.M > 0.01:
            self.m_gain = 1.0 / (1.0 + k * stimulus.M * t)
        if stimulus.S > 0.01:
            self.s_gain = 1.0 / (1.0 + k * stimulus.S * t)
        
        # Normalize to prevent complete loss
        max_gain = max(self.l_gain, self.m_gain, self.s_gain)
        if max_gain < 0.2:
            factor = 0.2 / max_gain
            self.l_gain *= factor
            self.m_gain *= factor
            self.s_gain *= factor
    
    def perceive(self, stimulus: ConeResponse) -> ConeResponse:
        """Apply current adaptation state to a stimulus."""
        return ConeResponse(
            L=stimulus.L * self.l_gain,
            M=stimulus.M * self.m_gain,
            S=stimulus.S * self.s_gain
        )
    
    def get_effective_m_dominance(self, stimulus: ConeResponse) -> float:
        """Calculate M-dominance after adaptation."""
        perceived = self.perceive(stimulus)
        return perceived.m_dominance


# ============================================================================
# OLO CALCULATION ENGINE
# ============================================================================

class OloCalculator:
    """
    Calculates optimal colors and adaptation sequences for olo approximation.
    """
    
    def __init__(self):
        self.color_science = DisplayColorScience()
        self.adaptation = VonKriesAdaptation()
        
        # Find optimal base olo color
        self.olo_rgb, self.olo_m_ratio = DisplayColorScience.find_max_m_dominance_rgb()
        
        # Adaptation color (red to reduce L-cone sensitivity)
        self.adaptation_rgb = (230, 40, 40)
        
        # Background colors
        self.neutral_gray = (128, 128, 128)
        self.opponent_surround = (180, 110, 110)  # Reddish for M enhancement
        
    def get_olo_analysis(self) -> dict:
        """Get comprehensive analysis of the olo approximation."""
        resp = DisplayColorScience.rgb_to_lms(*self.olo_rgb)
        
        return {
            'rgb': self.olo_rgb,
            'lms': (resp.L, resp.M, resp.S),
            'chromaticity': resp.lms_chromaticity,
            'm_dominance': resp.m_dominance,
            'theoretical_pure_m': "LMS = (0, 1, 0) - IMPOSSIBLE on display",
        }
    
    def calculate_post_adaptation_boost(self, 
                                         adapt_duration: float = 30.0) -> float:
        """
        Calculate the effective M-dominance boost from adaptation.
        
        After adapting to red, L-cone sensitivity is reduced,
        which increases perceived M-dominance.
        """
        # Reset and apply adaptation
        self.adaptation.reset()
        adapt_response = DisplayColorScience.rgb_to_lms(*self.adaptation_rgb)
        self.adaptation.adapt(adapt_response, adapt_duration)
        
        # Calculate boosted perception of olo
        olo_response = DisplayColorScience.rgb_to_lms(*self.olo_rgb)
        boosted = self.adaptation.get_effective_m_dominance(olo_response)
        
        return boosted / self.olo_m_ratio  # Boost factor
    
    def generate_temporal_dither_sequence(self, 
                                           frames: int = 60,
                                           modulation_hz: float = 15.0) -> List[Tuple[int, int, int]]:
        """
        Generate a temporal dithering sequence that may create
        enhanced color perception through flicker fusion.
        
        The idea: rapidly alternating between colors might
        create unusual cone response patterns.
        """
        sequence = []
        base_r, base_g, base_b = self.olo_rgb
        
        for i in range(frames):
            t = i / 60.0  # Assume 60 FPS
            phase = 2 * np.pi * modulation_hz * t
            
            # Modulate around the base olo color
            # Keep it in the M-dominant region
            r = int(np.clip(base_r + 10 * np.sin(phase), 0, 255))
            g = int(np.clip(base_g + 15 * np.sin(phase + np.pi/3), base_g-20, 255))
            b = int(np.clip(base_b + 20 * np.cos(phase), base_b-30, 255))
            
            sequence.append((r, g, b))
        
        return sequence


# ============================================================================
# DISPLAY ENGINE (Pygame-based)
# ============================================================================

if HAS_PYGAME:
    class OloDisplayEngine:
        """
        Low-level display engine for olo simulation.
        Uses pygame for precise timing and pixel control.
        """
        
        def __init__(self, width: int = 1024, height: int = 768):
            pygame.init()
            
            self.width = width
            self.height = height
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("OLO - Novel Color Simulation")
            self.clock = pygame.time.Clock()
            
            # Calculator
            self.calc = OloCalculator()
            
            # Fonts
            self.font = pygame.font.SysFont('Arial', 24)
            self.small_font = pygame.font.SysFont('Arial', 16)
            
            # State
            self.mode = 'menu'
            self.start_time = 0
            self.frame_count = 0
            
            # Pre-generate temporal sequence
            self.temporal_sequence = self.calc.generate_temporal_dither_sequence(600)
            
            # Print analysis
            self._print_analysis()
        
        def _print_analysis(self):
            """Print color science analysis."""
            analysis = self.calc.get_olo_analysis()
            
            print("\n" + "="*70)
            print("OLO APPROXIMATION ANALYSIS")
            print("="*70)
            print(f"Optimal RGB:        {analysis['rgb']}")
            print(f"LMS response:       L={analysis['lms'][0]:.4f}, "
                  f"M={analysis['lms'][1]:.4f}, S={analysis['lms'][2]:.4f}")
            print(f"lms chromaticity:   l={analysis['chromaticity'][0]:.4f}, "
                  f"m={analysis['chromaticity'][1]:.4f}, s={analysis['chromaticity'][2]:.4f}")
            print(f"M-dominance ratio:  {analysis['m_dominance']:.4f}")
            print(f"\nTheoretical olo:    {analysis['theoretical_pure_m']}")
            
            boost = self.calc.calculate_post_adaptation_boost(30.0)
            print(f"\nPost-adaptation M-dominance boost: {boost:.2f}x")
            print("="*70 + "\n")
        
        def draw_centered_text(self, text: str, y: int, 
                               color: Tuple[int, int, int] = (255, 255, 255),
                               font: Optional[pygame.font.Font] = None):
            """Draw centered text."""
            if font is None:
                font = self.font
            surface = font.render(text, True, color)
            rect = surface.get_rect(center=(self.width // 2, y))
            self.screen.blit(surface, rect)
        
        def draw_fixation_cross(self, x: int, y: int, 
                                color: Tuple[int, int, int] = (0, 0, 0),
                                size: int = 15, thickness: int = 2):
            """Draw a fixation cross."""
            pygame.draw.line(self.screen, color, (x - size, y), (x + size, y), thickness)
            pygame.draw.line(self.screen, color, (x, y - size), (x, y + size), thickness)
        
        def draw_stimulus_circle(self, color: Tuple[int, int, int], radius: int = 150):
            """Draw the central stimulus circle."""
            center = (self.width // 2, self.height // 2)
            pygame.draw.circle(self.screen, color, center, radius)
        
        def draw_subpixel_pattern(self, radius: int = 150):
            """
            Create a subpixel-level dithering pattern.
            Attempts to exploit LCD subpixel layout for unusual color mixing.
            """
            cx, cy = self.width // 2, self.height // 2
            olo = self.calc.olo_rgb
            
            for y in range(cy - radius, cy + radius):
                for x in range(cx - radius, cx + radius):
                    dx, dy = x - cx, y - cy
                    if dx*dx + dy*dy <= radius*radius:
                        # Create checkerboard pattern at pixel level
                        if (x + y) % 2 == 0:
                            color = (olo[0], olo[1], olo[2])
                        else:
                            # Slight variation for dithering
                            color = (max(0, olo[0]-5), min(255, olo[1]+5), olo[2])
                        self.screen.set_at((x, y), color)
        
        def run_menu(self):
            """Display main menu."""
            self.screen.fill((40, 40, 40))
            
            self.draw_centered_text("OLO - Novel Color Simulator", 80, (255, 255, 255))
            self.draw_centered_text("Based on Fong et al. (2025)", 110, (180, 180, 180), self.small_font)
            
            y = 200
            instructions = [
                ("1", "Chromatic Adaptation Sequence (RECOMMENDED)"),
                ("2", "Static Olo Display"),
                ("3", "Temporal Flicker (15 Hz)"),
                ("4", "Subpixel Dither Pattern"),
                ("5", "Opponent Surround Effect"),
                ("6", "Full Combined Sequence"),
            ]
            
            for key, desc in instructions:
                self.draw_centered_text(f"[{key}] {desc}", y)
                y += 40
            
            self.draw_centered_text("ESC: Return to menu | Q: Quit", y + 50, (150, 150, 150), self.small_font)
            
            # Preview swatch
            preview_rect = pygame.Rect(self.width//2 - 75, y + 100, 150, 80)
            pygame.draw.rect(self.screen, self.calc.olo_rgb, preview_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), preview_rect, 2)
            self.draw_centered_text(f"Olo: RGB{self.calc.olo_rgb}", y + 200, (200, 200, 200), self.small_font)
        
        def run_adaptation(self):
            """
            Chromatic adaptation sequence:
            1. Instructions (3s)
            2. Red adaptation (30s) - reduces L-cone sensitivity
            3. Olo display (15s) - should appear more saturated
            4. Return to menu
            """
            elapsed = time.time() - self.start_time
            cx, cy = self.width // 2, self.height // 2
            
            if elapsed < 3:
                # Instructions
                self.screen.fill(self.calc.neutral_gray)
                self.draw_centered_text("Chromatic Adaptation Sequence", 200)
                self.draw_centered_text("Fixate on the center cross", 250)
                self.draw_centered_text("Stare at the RED for 30 seconds", 300)
                self.draw_fixation_cross(cx, cy)
                
            elif elapsed < 33:
                # Red adaptation phase
                self.screen.fill(self.calc.neutral_gray)
                self.draw_stimulus_circle(self.calc.adaptation_rgb)
                self.draw_fixation_cross(cx, cy, (255, 255, 255))
                remaining = 33 - int(elapsed)
                self.draw_centered_text(f"ADAPTING... {remaining}s", 80)
                
            elif elapsed < 48:
                # Olo display phase  
                self.screen.fill(self.calc.neutral_gray)
                self.draw_stimulus_circle(self.calc.olo_rgb)
                self.draw_fixation_cross(cx, cy)
                self.draw_centered_text("OLO - Notice the enhanced saturation", 80)
                self.draw_centered_text("(Compare to normal cyan)", 110, (180, 180, 180), self.small_font)
                
            else:
                self.mode = 'menu'
        
        def run_static(self):
            """Static olo display with gray surround."""
            self.screen.fill(self.calc.neutral_gray)
            self.draw_stimulus_circle(self.calc.olo_rgb)
            self.draw_fixation_cross(self.width//2, self.height//2)
            
            resp = DisplayColorScience.rgb_to_lms(*self.calc.olo_rgb)
            self.draw_centered_text(f"Static OLO: RGB{self.calc.olo_rgb}", 80)
            self.draw_centered_text(f"M-dominance: {resp.m_dominance:.4f}", 110, 
                                   (180, 180, 180), self.small_font)
        
        def run_flicker(self):
            """Temporal flicker mode."""
            self.screen.fill(self.calc.neutral_gray)
            
            idx = self.frame_count % len(self.temporal_sequence)
            color = self.temporal_sequence[idx]
            
            self.draw_stimulus_circle(color)
            self.draw_fixation_cross(self.width//2, self.height//2)
            self.draw_centered_text("Temporal Flicker (15 Hz)", 80)
        
        def run_subpixel(self):
            """Subpixel dithering pattern."""
            self.screen.fill(self.calc.neutral_gray)
            self.draw_subpixel_pattern()
            self.draw_fixation_cross(self.width//2, self.height//2)
            self.draw_centered_text("Subpixel Dither Pattern", 80)
        
        def run_opponent(self):
            """Opponent color surround."""
            self.screen.fill(self.calc.opponent_surround)
            self.draw_stimulus_circle(self.calc.olo_rgb)
            self.draw_fixation_cross(self.width//2, self.height//2)
            self.draw_centered_text("Opponent Surround (reddish background)", 80)
        
        def run_full_sequence(self):
            """Combined techniques."""
            elapsed = time.time() - self.start_time
            cx, cy = self.width // 2, self.height // 2
            
            # Timeline
            phases = [
                (0, 3, "Prepare: Fixate on center"),
                (3, 23, "Phase 1: Adaptation (20s)"),
                (23, 33, "Phase 2: Olo"),
                (33, 38, "Phase 3: Opponent Surround"),
                (38, 48, "Phase 4: Temporal Flicker"),
                (48, 53, "Phase 5: Final Olo"),
            ]
            
            # Find current phase
            current_phase = None
            for start, end, name in phases:
                if start <= elapsed < end:
                    current_phase = (start, end, name)
                    break
            
            if current_phase is None:
                self.mode = 'menu'
                return
            
            start, end, name = current_phase
            
            if "Prepare" in name:
                self.screen.fill((40, 40, 40))
                self.draw_centered_text(name, cy)
            elif "Adaptation" in name:
                self.screen.fill(self.calc.neutral_gray)
                self.draw_stimulus_circle(self.calc.adaptation_rgb)
                self.draw_fixation_cross(cx, cy, (255, 255, 255))
            elif "Opponent" in name:
                self.screen.fill(self.calc.opponent_surround)
                self.draw_stimulus_circle(self.calc.olo_rgb)
                self.draw_fixation_cross(cx, cy)
            elif "Flicker" in name:
                self.screen.fill(self.calc.neutral_gray)
                idx = self.frame_count % len(self.temporal_sequence)
                self.draw_stimulus_circle(self.temporal_sequence[idx])
                self.draw_fixation_cross(cx, cy)
            else:  # Olo phases
                self.screen.fill(self.calc.neutral_gray)
                self.draw_stimulus_circle(self.calc.olo_rgb)
                self.draw_fixation_cross(cx, cy)
            
            self.draw_centered_text(name, 80)
        
        def run(self):
            """Main loop."""
            running = True
            
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                        elif event.key == pygame.K_ESCAPE:
                            self.mode = 'menu'
                        elif self.mode == 'menu':
                            if event.key == pygame.K_1:
                                self.mode = 'adaptation'
                                self.start_time = time.time()
                            elif event.key == pygame.K_2:
                                self.mode = 'static'
                            elif event.key == pygame.K_3:
                                self.mode = 'flicker'
                            elif event.key == pygame.K_4:
                                self.mode = 'subpixel'
                            elif event.key == pygame.K_5:
                                self.mode = 'opponent'
                            elif event.key == pygame.K_6:
                                self.mode = 'sequence'
                                self.start_time = time.time()
                
                # Render current mode
                if self.mode == 'menu':
                    self.run_menu()
                elif self.mode == 'adaptation':
                    self.run_adaptation()
                elif self.mode == 'static':
                    self.run_static()
                elif self.mode == 'flicker':
                    self.run_flicker()
                elif self.mode == 'subpixel':
                    self.run_subpixel()
                elif self.mode == 'opponent':
                    self.run_opponent()
                elif self.mode == 'sequence':
                    self.run_full_sequence()
                
                pygame.display.flip()
                self.clock.tick(60)
                self.frame_count += 1
            
            pygame.quit()


# ============================================================================
# ANALYSIS-ONLY MODE (No display required)
# ============================================================================

def run_analysis_only():
    """Run color science analysis without display."""
    print("\n" + "="*70)
    print("OLO COLOR SCIENCE ANALYSIS")
    print("="*70)
    
    # Find optimal olo RGB
    optimal_rgb, m_ratio = DisplayColorScience.find_max_m_dominance_rgb()
    
    print(f"\n[1] OPTIMAL OLO APPROXIMATION")
    print(f"    RGB: {optimal_rgb}")
    print(f"    M-dominance ratio: {m_ratio:.4f}")
    
    resp = DisplayColorScience.rgb_to_lms(*optimal_rgb)
    print(f"    LMS: ({resp.L:.4f}, {resp.M:.4f}, {resp.S:.4f})")
    print(f"    lms chromaticity: {resp.lms_chromaticity}")
    
    print(f"\n[2] THEORETICAL PURE OLO (IMPOSSIBLE)")
    print(f"    LMS: (0, 1, 0)")
    print(f"    M-dominance ratio: INFINITE")
    print(f"    This requires M-cone-only stimulation")
    
    print(f"\n[3] COMPARISON WITH STANDARD COLORS")
    comparisons = [
        ("Pure Cyan (0,255,255)", (0, 255, 255)),
        ("Pure Green (0,255,0)", (0, 255, 0)),
        ("507nm approx (0,219,182)", (0, 219, 182)),
        ("Optimal Olo", optimal_rgb),
    ]
    
    print(f"    {'Color':<30} {'RGB':<20} {'M/(L+S)':<10}")
    print(f"    {'-'*28} {'-'*18} {'-'*10}")
    
    for name, rgb in comparisons:
        r = DisplayColorScience.rgb_to_lms(*rgb)
        print(f"    {name:<30} {str(rgb):<20} {r.m_dominance:.4f}")
    
    print(f"\n[4] CHROMATIC ADAPTATION EFFECT")
    calc = OloCalculator()
    boost = calc.calculate_post_adaptation_boost(30.0)
    print(f"    After 30s red adaptation:")
    print(f"    L-cone sensitivity reduced")
    print(f"    Effective M-dominance boost: {boost:.2f}x")
    
    print(f"\n[5] PAPER REFERENCE (Fong et al. 2025)")
    print(f"    Subject descriptions of olo:")
    print(f"    - 'teal'")
    print(f"    - 'blue-green of unprecedented saturation'")
    print(f"    - 'green, a little blue'")
    print(f"    - Saturation rating: 4/4 (maximum)")
    print(f"    - Matched to 501-512nm (but required desaturation)")
    
    print("\n" + "="*70)
    print("KEY INSIGHT: True olo is OUTSIDE the natural color gamut.")
    print("This simulator can only approximate it perceptually.")
    print("="*70 + "\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    print("\nOLO - Novel Color Simulator")
    print("Based on: Fong et al. (2025) Science Advances")
    print("-" * 50)
    
    if not HAS_PYGAME:
        print("\nPygame not available. Running analysis only.")
        run_analysis_only()
        return
    
    print("\nPygame available. Launching display...")
    print("\nControls:")
    print("  1: Chromatic Adaptation (recommended)")
    print("  2: Static Olo")
    print("  3: Temporal Flicker")
    print("  4: Subpixel Pattern")
    print("  5: Opponent Surround")
    print("  6: Full Sequence")
    print("  ESC: Menu | Q: Quit")
    print("-" * 50)
    
    engine = OloDisplayEngine()
    engine.run()


if __name__ == "__main__":
    main()