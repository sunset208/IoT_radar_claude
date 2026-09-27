import sys
import threading
import pygame
import pygame.freetype
import numpy as np
import matplotlib.pyplot as plt
from MicroDopplerDetection.main import main

# =========================================================
# INIT
# =========================================================
pygame.init()
pygame.freetype.init()

WIDTH, HEIGHT = 1200, 760

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.SCALED | pygame.DOUBLEBUF
)

pygame.display.set_caption("Radar Modulaire")

clock = pygame.time.Clock()
FPS = 60

# =========================================================
# THEMES
# =========================================================

THEMES = {
    "dark": {
        "bg": (8, 10, 18),
        "card": (18, 22, 34),
        "card_2": (26, 30, 46),
        "text": (240, 245, 255),
        "text_secondary": (145, 155, 180),
        "accent": (0, 220, 255),
        "accent_2": (170, 120, 255),
        "button": (30, 35, 50),
        "button_hover": (45, 55, 78),
        "border": (60, 90, 140),
        "shadow": (0, 0, 0, 140),
        "badge": (0, 220, 255)
    },
    "light": {
        "bg": (240, 244, 252),
        "card": (255, 255, 255),
        "card_2": (245, 248, 255),
        "text": (20, 24, 35),
        "text_secondary": (90, 100, 125),
        "accent": (0, 140, 255),
        "accent_2": (120, 90, 255),
        "button": (228, 234, 246),
        "button_hover": (210, 220, 240),
        "border": (180, 190, 220),
        "shadow": (0, 0, 0, 30),
        "badge": (0, 140, 255)
    }
}

theme_mode = "light"

# =========================================================
# FONTS
# =========================================================

try:
    title_font = pygame.freetype.SysFont("Segoe UI Semibold", 44)
    subtitle_font = pygame.freetype.SysFont("Segoe UI", 20)
    button_font = pygame.freetype.SysFont("Segoe UI Semibold", 18)
    badge_font = pygame.freetype.SysFont("Segoe UI Bold", 15)
    small_font = pygame.freetype.SysFont("Segoe UI", 14)
except:
    title_font = pygame.freetype.SysFont("Arial", 44)
    subtitle_font = pygame.freetype.SysFont("Arial", 20)
    button_font = pygame.freetype.SysFont("Arial", 18)
    badge_font = pygame.freetype.SysFont("Arial", 15)
    small_font = pygame.freetype.SysFont("Arial", 14)

# =========================================================
# STATES
# =========================================================

etat_ecran = "principal"
btn_acquisition_pressed = False  # État pour l'effet "ouvert" du bouton Acquisition

# =========================================================
# UTILS
# =========================================================

def current_theme():
    return THEMES[theme_mode]

def draw_rounded_rect(surface, rect, color, radius=20):
    pygame.draw.rect(surface, color, rect, border_radius=radius)

def draw_shadow(surface, rect, shadow_color):
    shadow_rect = rect.copy()
    shadow_rect.x += 6
    shadow_rect.y += 8
    shadow_surface = pygame.Surface(
        (shadow_rect.width, shadow_rect.height),
        pygame.SRCALPHA
    )
    pygame.draw.rect(
        shadow_surface,
        shadow_color,
        (0, 0, shadow_rect.width, shadow_rect.height),
        border_radius=24
    )
    surface.blit(shadow_surface, shadow_rect)

def draw_card(surface, rect):
    theme = current_theme()
    draw_shadow(surface, rect, theme["shadow"])
    draw_rounded_rect(surface, rect, theme["card"], 24)
    pygame.draw.rect(
        surface,
        theme["border"],
        rect,
        width=1,
        border_radius=24
    )

def draw_button(rect, text, hovered=False, accent=False, pressed=False):
    theme = current_theme()
    bg = theme["button_hover"] if hovered else theme["button"]
    if accent:
        bg = theme["accent"]
    if pressed:  # Effet "ouvert" si le bouton est pressé
        bg = theme["accent_2"]
    draw_rounded_rect(screen, rect, bg, 16)
    pygame.draw.rect(
        screen,
        theme["border"],
        rect,
        width=1,
        border_radius=16
    )
    text_color = (255, 255, 255) if accent or pressed else theme["text"]
    surf, txt_rect = button_font.render(text, text_color)
    txt_rect.center = rect.center
    screen.blit(surf, txt_rect)

def draw_badge(center_x, y, text):
    theme = current_theme()
    badge_rect = pygame.Rect(0, 0, 190, 38)
    badge_rect.center = (center_x, y)
    glow = pygame.Surface((210, 58), pygame.SRCALPHA)
    pygame.draw.rect(
        glow,
        (*theme["badge"], 40),
        (0, 0, 210, 58),
        border_radius=22
    )
    screen.blit(glow, (badge_rect.x - 10, badge_rect.y - 10))
    draw_rounded_rect(screen, badge_rect, theme["card_2"], 18)
    pygame.draw.rect(
        screen,
        theme["accent"],
        badge_rect,
        width=1,
        border_radius=18
    )
    surf, txt_rect = badge_font.render(text, theme["accent"])
    txt_rect.center = badge_rect.center
    screen.blit(surf, txt_rect)

# =========================================================
# MATPLOTLIB
# =========================================================

def courbe_temps_reel():
    theme = current_theme()
    plt.style.use("dark_background" if theme_mode == "dark" else "default")
    fig = plt.figure(
        "Courbe Temps Réel",
        facecolor=np.array(theme["bg"]) / 255
    )
    ax = fig.add_subplot(111)
    ax.set_facecolor(np.array(theme["bg"]) / 255)
    color = np.array(theme["accent"]) / 255
    x_data = []
    y_data = []
    for i in range(2220):
        x_data.append(i)
        y_data.append(np.sin(i * 0.1))
        ax.clear()
        ax.plot(
            x_data,
            y_data,
            color=color,
            linewidth=2.8
        )
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(
            "Signal Temps Réel",
            color=color,
            fontsize=16
        )
        ax.grid(alpha=0.25)
        plt.pause(0.001)
        if not plt.fignum_exists("Courbe Temps Réel"):
            break
    plt.close()

# =========================================================
# BUTTONS
# =========================================================

BTN_W = 260
BTN_H = 54
BTN_SPACE = 18

center_x = WIDTH // 2

btn_acquisition = pygame.Rect(0, 0, BTN_W, BTN_H)
btn_quit = pygame.Rect(0, 0, BTN_W, BTN_H)
btn_theme = pygame.Rect(0, 0, BTN_W, BTN_H)

# =========================================================
# MAIN LOOP
# =========================================================

running = True

while running:
    mouse_pos = pygame.mouse.get_pos()
    theme = current_theme()

    # =====================================================
    # EVENTS
    # =====================================================

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if btn_acquisition.collidepoint(mouse_pos):
                btn_acquisition_pressed = True
                threading.Thread(
                    target=main,
                    daemon=True
                ).start()
            elif btn_quit.collidepoint(mouse_pos):
                running = False
            elif btn_theme.collidepoint(mouse_pos):
                theme_mode = (
                    "light"
                    if theme_mode == "dark"
                    else "dark"
                )

        if event.type == pygame.MOUSEBUTTONUP:
            if btn_acquisition_pressed:
                btn_acquisition_pressed = False

    # =====================================================
    # BACKGROUND
    # =====================================================

    screen.fill(theme["bg"])

    # =====================================================
    # MAIN SCREEN
    # =====================================================

    card_rect = pygame.Rect(0, 0, 520, 560)
    card_rect.center = (WIDTH // 2, HEIGHT // 2)

    draw_card(screen, card_rect)

    # =================================================
    # TITLE
    # =================================================

    title_surf, title_rect = title_font.render(
        "Radar Modulaire",
        theme["text"]
    )
    title_rect.center = (
        WIDTH // 2,
        card_rect.y + 70
    )
    screen.blit(title_surf, title_rect)

    # =================================================
    # SUBTITLE
    # =================================================

    subtitle_surf, subtitle_rect = subtitle_font.render(
        "",
        theme["text_secondary"]
    )
    subtitle_rect.center = (
        WIDTH // 2,
        title_rect.bottom + 30
    )
    screen.blit(subtitle_surf, subtitle_rect)

    # =================================================
    # BADGE
    # =================================================

    draw_badge(
        WIDTH // 2,
        subtitle_rect.bottom + 45,
        "Projet S6 - Pôle IoT"
    )

    # =================================================
    # BUTTONS
    # =================================================

    start_y = card_rect.y + 260

    btn_acquisition.center = (center_x, start_y)
    btn_theme.center = (center_x, start_y + BTN_H + BTN_SPACE)  # Mode sombre/clair en 2ème position
    btn_quit.center = (center_x, start_y + (BTN_H + BTN_SPACE) * 2)  # Quitter en 3ème position

    draw_button(
        btn_acquisition,
        "Lancer Acquisition",
        btn_acquisition.collidepoint(mouse_pos),
        accent=True,
        pressed=btn_acquisition_pressed  # Effet "ouvert" si pressé
    )

    draw_button(
        btn_theme,
        "Mode sombre / clair",
        btn_theme.collidepoint(mouse_pos)
    )

    draw_button(
        btn_quit,
        "Quitter",
        btn_quit.collidepoint(mouse_pos)
    )

    # =================================================
    # FOOTER
    # =================================================

    footer_surf, footer_rect = small_font.render(
        "2026",
        theme["text_secondary"]
    )
    footer_rect.center = (
        WIDTH // 2,
        card_rect.bottom - 40
    )
    screen.blit(footer_surf, footer_rect)

    # =====================================================
    # DISPLAY
    # =====================================================

    pygame.display.flip()
    clock.tick(FPS)

# =========================================================
# QUIT
# =========================================================

pygame.quit()
sys.exit()