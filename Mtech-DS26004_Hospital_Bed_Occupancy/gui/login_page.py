"""Secure demo login backed by salted PBKDF2 hashes in SQLite."""
from __future__ import annotations
import hashlib
import secrets
import sqlite3
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable

from src.utils import ROOT
from .theme import COLORS


class CredentialStore:
    """Minimal local credential store without plaintext passwords."""
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or ROOT / "database/users.db"; self.path.parent.mkdir(parents=True, exist_ok=True); self._initialize()

    @staticmethod
    def _hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 240_000)

    def _initialize(self) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, salt BLOB NOT NULL, password_hash BLOB NOT NULL)")
            if not connection.execute("SELECT 1 FROM users WHERE username=?", ("admin",)).fetchone():
                salt = secrets.token_bytes(16); connection.execute("INSERT INTO users VALUES(?,?,?)", ("admin", salt, self._hash("Hospital@123", salt)))

    def verify(self, username: str, password: str) -> bool:
        with sqlite3.connect(self.path) as connection: row = connection.execute("SELECT salt,password_hash FROM users WHERE username=?", (username,)).fetchone()
        return bool(row and secrets.compare_digest(self._hash(password, row[0]), row[1]))


class LoginPage(ttk.Frame):
    """Professional authentication screen."""
    def __init__(self, parent: tk.Widget, on_login: Callable[[], None]) -> None:
        super().__init__(parent); self.on_login, self.store = on_login, CredentialStore(); self.columnconfigure(0, weight=1); self.rowconfigure(0, weight=1)
        card = ttk.Frame(self, style="Card.TFrame", padding=38); card.grid(row=0, column=0, padx=20, pady=20)
        tk.Canvas(card, width=72, height=72, bg=COLORS["card"], highlightthickness=0).pack()
        logo = card.winfo_children()[0]; logo.create_oval(4, 4, 68, 68, fill=COLORS["blue"], outline=""); logo.create_text(36, 36, text="H+", fill="white", font=("Segoe UI Semibold", 24))
        ttk.Label(card, text="HarborBed Analytics", style="KPI.TLabel").pack(pady=(8, 2)); ttk.Label(card, text="Hospital occupancy repair & risk intelligence", style="Card.TLabel").pack(pady=(0, 20))
        self.username = tk.StringVar(value="admin"); self.password = tk.StringVar(value="Hospital@123"); self.message = tk.StringVar()
        ttk.Label(card, text="Username", style="Card.TLabel").pack(anchor="w"); ttk.Entry(card, textvariable=self.username, width=36).pack(fill="x", pady=(3, 10))
        ttk.Label(card, text="Password", style="Card.TLabel").pack(anchor="w"); self.password_entry = ttk.Entry(card, textvariable=self.password, show="•"); self.password_entry.pack(fill="x", pady=(3, 5))
        self.show = tk.BooleanVar(); ttk.Checkbutton(card, text="Show password", variable=self.show, command=lambda: self.password_entry.configure(show="" if self.show.get() else "•")).pack(anchor="w")
        ttk.Label(card, textvariable=self.message, foreground=COLORS["red"], style="Card.TLabel").pack(pady=8)
        buttons = ttk.Frame(card, style="Card.TFrame"); buttons.pack(fill="x"); ttk.Button(buttons, text="Login", style="Accent.TButton", command=self.login).pack(side="left", expand=True, fill="x", padx=(0, 5)); ttk.Button(buttons, text="Reset", command=self.reset).pack(side="left", expand=True, fill="x")
        ttk.Label(card, text="Demo account: admin / Hospital@123", style="Card.TLabel").pack(pady=(16, 0)); self.password_entry.bind("<Return>", lambda _e: self.login())

    def login(self) -> None:
        if self.store.verify(self.username.get().strip(), self.password.get()): self.message.set(""); self.on_login()
        else: self.message.set("Invalid username or password.")

    def reset(self) -> None:
        self.username.set(""); self.password.set(""); self.message.set("")
