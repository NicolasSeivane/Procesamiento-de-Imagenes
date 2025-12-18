"""
Procesamiento de Imágenes - Main Entry Point
============================================

Advanced Image Processing Application with Tkinter GUI
Implements various algorithms for filtering, edge detection,
feature extraction, and noise analysis.

Author: Nicolas Seivane
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.gui.tkinter_app import root

if __name__ == "__main__":
    print("=" * 60)
    print("Procesamiento de Imágenes - Image Processing Application")
    print("=" * 60)
    print("Starting GUI...")
    root.mainloop()
