"""Launch the HarborBed Analytics desktop application."""
from __future__ import annotations
from gui.application import HospitalOccupancyApp


def main() -> None:
    app = HospitalOccupancyApp()
    app.mainloop()


if __name__ == "__main__":
    main()
