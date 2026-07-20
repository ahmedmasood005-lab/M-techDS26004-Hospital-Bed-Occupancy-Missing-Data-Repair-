"""Reusable modern GUI components."""
from __future__ import annotations
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from .theme import COLORS


class Tooltip:
    """Small delayed hover tooltip."""
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show, add=True); widget.bind("<Leave>", self.hide, add=True)

    def show(self, _event: tk.Event[Any]) -> None:
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget); self.tip.overrideredirect(True); self.tip.geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, bg=COLORS["navy"], fg="white", padx=8, pady=5, font=("Segoe UI", 9)).pack()

    def hide(self, _event: tk.Event[Any] | None = None) -> None:
        if self.tip: self.tip.destroy(); self.tip = None


class ScrollablePage(ttk.Frame):
    """Base page with vertically scrollable content and a stable header."""
    def __init__(self, parent: tk.Widget, app: Any, title: str, subtitle: str) -> None:
        super().__init__(parent); self.app = app
        ttk.Label(self, text=title, style="Title.TLabel").pack(anchor="w", padx=24, pady=(20, 2))
        ttk.Label(self, text=subtitle, style="Subtitle.TLabel").pack(anchor="w", padx=24, pady=(0, 12))
        shell = ttk.Frame(self); shell.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        self.canvas = tk.Canvas(shell, bg=COLORS["bg"], highlightthickness=0); bar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas); self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set); self.canvas.pack(side="left", fill="both", expand=True); bar.pack(side="right", fill="y")
        self.content.bind("<Configure>", lambda _e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self.window, width=e.width))

    def refresh(self) -> None:
        """Update page content when navigated to."""
        self.canvas.yview_moveto(0)


class KPI(ttk.Frame):
    """Reusable colored metric card."""
    def __init__(self, parent: tk.Widget, label: str, value: str, color: str) -> None:
        super().__init__(parent, style="Card.TFrame", padding=14); self.configure(width=180, height=90); self.pack_propagate(False)
        tk.Frame(self, bg=color, width=5).pack(side="left", fill="y", padx=(0, 10))
        body = ttk.Frame(self, style="Card.TFrame"); body.pack(fill="both", expand=True)
        ttk.Label(body, text=label, style="Card.TLabel").pack(anchor="w"); self.value = ttk.Label(body, text=value, style="KPI.TLabel"); self.value.pack(anchor="w", pady=(6, 0))


def dataframe_tree(parent: tk.Widget, frame: Any, height: int = 12) -> ttk.Treeview:
    """Create a scrollable Treeview preview for a DataFrame."""
    wrapper = ttk.Frame(parent); wrapper.pack(fill="both", expand=True)
    columns = [str(c) for c in frame.columns]; tree = ttk.Treeview(wrapper, columns=columns, show="headings", height=height)
    ybar = ttk.Scrollbar(wrapper, orient="vertical", command=tree.yview); xbar = ttk.Scrollbar(wrapper, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    for column in columns:
        tree.heading(column, text=column, command=lambda c=column: sort_tree(tree, c, False)); tree.column(column, width=130, minwidth=80, anchor="center")
    for row in frame.head(250).itertuples(index=False, name=None): tree.insert("", "end", values=["" if str(v) == "nan" else str(v)[:32] for v in row])
    tree.grid(row=0, column=0, sticky="nsew"); ybar.grid(row=0, column=1, sticky="ns"); xbar.grid(row=1, column=0, sticky="ew"); wrapper.rowconfigure(0, weight=1); wrapper.columnconfigure(0, weight=1)
    return tree


def sort_tree(tree: ttk.Treeview, column: str, reverse: bool) -> None:
    """Sort Treeview values numerically when possible."""
    values = [(tree.set(item, column), item) for item in tree.get_children("")]
    def key(pair: tuple[str, str]) -> tuple[int, Any]:
        try: return (0, float(pair[0]))
        except ValueError: return (1, pair[0].lower())
    values.sort(key=key, reverse=reverse)
    for index, (_, item) in enumerate(values): tree.move(item, "", index)
    tree.heading(column, command=lambda: sort_tree(tree, column, not reverse))


def run_async(app: Any, task: Callable[[], Any], on_success: Callable[[Any], None], label: str) -> None:
    """Run expensive work in a daemon thread and marshal updates to Tk."""
    app.set_status(label); app.progress.start(12)
    def worker() -> None:
        try:
            result = task(); app.after(0, lambda: on_success(result)); app.after(0, lambda: app.set_status(f"Completed: {label}"))
        except Exception as exc:
            app.after(0, lambda: app.show_error(str(exc)))
        finally:
            app.after(0, app.progress.stop)
    threading.Thread(target=worker, daemon=True).start()
