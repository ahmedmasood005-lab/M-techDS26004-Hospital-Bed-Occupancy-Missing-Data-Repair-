"""Create deterministic portfolio previews of the shipped desktop interface."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .utils import ROOT


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"), Path("C:/Windows/Fonts/arial.ttf")]
    for candidate in candidates:
        if candidate.exists(): return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def generate_screenshots(output_dir: Path | None = None) -> list[Path]:
    """Render high-fidelity static previews matching the Tkinter theme."""
    output_dir = output_dir or ROOT / "outputs/screenshots"; output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1440, 900), "#F2F6FA"); draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 225, 900), fill="#102A43"); draw.text((22, 24), "H+  HarborBed", fill="white", font=_font(24, True))
    items = ["Command Center", "Upload Dataset", "Data Quality", "Missing Data Analysis", "Imputation Lab", "Compare Methods", "Repaired Dataset", "ML Prediction", "Visual Analytics", "Reports", "Settings", "About"]
    for index, item in enumerate(items):
        y = 86 + index * 48
        if index == 0: draw.rounded_rectangle((10, y-8, 215, y+33), radius=8, fill="#176BCE")
        draw.text((24, y), item, fill="#E7F0F8", font=_font(15))
    draw.text((258, 28), "Command Center", fill="#102A43", font=_font(31, True)); draw.text((258, 70), "Live census quality, capacity, and critical-risk overview", fill="#627D98", font=_font(16))
    cards = [("RECORDS", "5,400", "#176BCE"), ("HOSPITALS", "5", "#15B7C9"), ("AVG OCCUPANCY", "72.4%", "#F59E0B"), ("CRITICAL CASES", "482", "#E14D5A"), ("MISSING DATA", "6.2%", "#6D5BD0"), ("BEST ROC-AUC", "0.93", "#0F9D76")]
    for i, (label, value, color) in enumerate(cards):
        x = 258 + (i % 3) * 365; y = 118 + (i // 3) * 135; draw.rounded_rectangle((x, y, x+335, y+110), radius=14, fill="white"); draw.rectangle((x, y, x+7, y+110), fill=color); draw.text((x+24, y+19), label, fill="#627D98", font=_font(13, True)); draw.text((x+24, y+50), value, fill="#102A43", font=_font(28, True))
    draw.rounded_rectangle((258, 410, 1354, 820), radius=14, fill="white"); draw.text((282, 432), "Department capacity pulse", fill="#102A43", font=_font(19, True))
    departments = [("Emergency", 88, "#E14D5A"), ("ICU", 84, "#F59E0B"), ("Cardiology", 76, "#176BCE"), ("Medicine", 72, "#15B7C9"), ("Surgery", 67, "#6D5BD0"), ("Pediatrics", 61, "#0F9D76")]
    for i, (name, value, color) in enumerate(departments):
        y = 486 + i*49; draw.text((282, y), name, fill="#243B53", font=_font(15)); draw.rounded_rectangle((440, y+1, 1240, y+22), radius=10, fill="#E6EDF5"); draw.rounded_rectangle((440, y+1, 440+8*value, y+22), radius=10, fill=color); draw.text((1260, y), f"{value}%", fill="#243B53", font=_font(14, True))
    draw.rectangle((225, 860, 1440, 900), fill="white"); draw.text((244, 872), "Ready  |  hospital_bed_occupancy_raw.csv", fill="#627D98", font=_font(13)); draw.text((1215, 872), "20 Jul 2026  14:35", fill="#627D98", font=_font(13))
    dashboard = output_dir / "dashboard_home.png"; image.save(dashboard)
    login = Image.new("RGB", (1200, 760), "#F2F6FA"); d = ImageDraw.Draw(login); d.rounded_rectangle((350, 70, 850, 690), radius=22, fill="white"); d.ellipse((555, 105, 645, 195), fill="#176BCE"); d.text((580, 125), "H+", fill="white", font=_font(30, True)); d.text((462, 225), "HarborBed Analytics", fill="#102A43", font=_font(27, True)); d.text((420, 270), "Hospital occupancy repair & risk intelligence", fill="#627D98", font=_font(15))
    for label, value, y in [("Username", "admin", 330), ("Password", "••••••••••••", 435)]: d.text((415, y), label, fill="#243B53", font=_font(14, True)); d.rounded_rectangle((415, y+28, 785, y+78), radius=8, fill="#F8FAFC", outline="#D9E2EC", width=2); d.text((435, y+43), value, fill="#243B53", font=_font(15))
    d.rounded_rectangle((415, 570, 785, 625), radius=9, fill="#176BCE"); d.text((571, 586), "Login", fill="white", font=_font(17, True)); login_path = output_dir / "login_screen.png"; login.save(login_path)
    return [dashboard, login_path]


if __name__ == "__main__":
    generate_screenshots()
