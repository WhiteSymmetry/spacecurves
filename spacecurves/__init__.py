# -*- coding: utf-8 -*-
# __init__.py

"""
Space Curves (spacecurves, Uzay Eğrileri):
spacecurves.py - Uzay Dolduran Eğriler Modülü

Bu modül, Hilbert eğrisi ve diğer uzay dolduran eğriler için
tüm özellikleri içeren eksiksiz bir implementasyon sunar.

Özellikler:
- 1-10 derinlik (2^p grid)
- 1-5 boyut desteği
- 4 farklı eğri tipi (Hilbert, Morton, Moore, Altair)
- Önbellek desteği
- Batch işlemler
- Komşuluk analizi
- Görüntü sıkıştırma
- Boyut indirgeme
- Hilbert sıralama
- Grid sistemi
- Rota optimizasyonu
"""

from __future__ import annotations
import warnings

# Paket sürüm numarası
__version__ = "0.1.1"
__author__ = "Mehmet Keçeci"
__email__ = "mkececi@yaani.com"
__description__ = "Space Curves (spacecurves, Uzay Eğrileri): Uzay Dolduran Eğriler Modülü"

# Ana Sınıflar
from .spacecurves import (
    SpaceFillingCurve,
    CurveType,
    CurveStats,
    HilbertImageCompressor,
    HilbertDimensionReducer,
    HilbertOrdering,
    HilbertGrid,
    HilbertPathOptimizer
)

# Public API - from * import ile erişilebilenler
__all__ = [
    # Ana sınıflar
    'SpaceFillingCurve',
    'CurveType',
    'CurveStats',
    
    # Yardımcı sınıflar
    'HilbertImageCompressor',
    'HilbertDimensionReducer',
    'HilbertOrdering',
    'HilbertGrid',
    'HilbertPathOptimizer',
]

# Paket yüklendiğinde kısa bilgi
def _show_welcome():
    print(f"🌟 Space Curves (spacecurves, Uzay Eğrileri) v{__version__} yüklendi")
    print(f"   📌 Kullanım: from spacecurves import SpaceFillingCurve, CurveType")
    print(f"   📌 Örnek: curve = SpaceFillingCurve(p=4, n=2)")

# Otomatik welcome mesajı (isteğe bağlı)
# _show_welcome()
