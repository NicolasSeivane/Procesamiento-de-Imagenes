"""
Procesamiento de Imágenes - Image Processing Library
====================================================

A comprehensive image processing library implementing various algorithms
for filtering, edge detection, feature extraction, and noise analysis.

Modules:
    - core: Core image processing algorithms
    - gui: Graphical user interfaces
    - utils: Utility functions
"""

__version__ = "1.0.0"
__author__ = "Nicolas Seivane"

from src.core import image_processing, sift
from src.utils import noise_generator

__all__ = ['image_processing', 'sift', 'noise_generator']
