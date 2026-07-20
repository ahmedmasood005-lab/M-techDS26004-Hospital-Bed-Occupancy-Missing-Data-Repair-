"""Application design tokens and ttk theme construction."""
from __future__ import annotations
import tkinter as tk
from tkinter import ttk

COLORS = {"navy": "#102A43", "navy2": "#183B56", "blue": "#176BCE", "cyan": "#15B7C9",
          "purple": "#6D5BD0", "green": "#0F9D76", "orange": "#F59E0B", "red": "#E14D5A",
          "bg": "#F2F6FA", "card": "#FFFFFF", "text": "#243B53", "muted": "#627D98", "line": "#D9E2EC"}


def apply_theme(root: tk.Tk, dark: bool = False, font_size: int = 10) -> None:
    """Apply the custom Harbor Analytics ttk theme."""
    style = ttk.Style(root); style.theme_use("clam")
    background = COLORS["navy"] if dark else COLORS["bg"]
    foreground = "#E6EDF5" if dark else COLORS["text"]
    card = COLORS["navy2"] if dark else COLORS["card"]
    root.configure(bg=background)
    style.configure(".", font=("Segoe UI", font_size), background=background, foreground=foreground)
    style.configure("TFrame", background=background); style.configure("Card.TFrame", background=card, relief="flat")
    style.configure("TLabel", background=background, foreground=foreground); style.configure("Card.TLabel", background=card, foreground=foreground)
    style.configure("Title.TLabel", font=("Segoe UI Semibold", font_size + 12), foreground=COLORS["navy"] if not dark else "white")
    style.configure("Subtitle.TLabel", font=("Segoe UI", font_size), foreground=COLORS["muted"])
    style.configure("KPI.TLabel", font=("Segoe UI Semibold", font_size + 8), background=card, foreground=COLORS["navy"] if not dark else "white")
    style.configure("Accent.TButton", background=COLORS["blue"], foreground="white", padding=(16, 9), borderwidth=0)
    style.map("Accent.TButton", background=[("active", COLORS["cyan"]), ("disabled", COLORS["muted"] )])
    style.configure("Danger.TButton", background=COLORS["red"], foreground="white", padding=(14, 8))
    style.configure("Treeview", background=card, fieldbackground=card, foreground=foreground, rowheight=28, borderwidth=0)
    style.configure("Treeview.Heading", background=COLORS["navy2"], foreground="white", font=("Segoe UI Semibold", font_size), padding=7)
    style.map("Treeview", background=[("selected", COLORS["blue"])], foreground=[("selected", "white")])
    style.configure("TEntry", fieldbackground=card, padding=7); style.configure("TCombobox", fieldbackground=card, padding=5)
    style.configure("Horizontal.TProgressbar", troughcolor=COLORS["line"], background=COLORS["cyan"], thickness=10)
