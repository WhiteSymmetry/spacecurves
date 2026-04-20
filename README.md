# Space Curves (spacecurves, Uzay Eğrileri) <img src="docs/logo.jpg" alt="Space Curves (spacecurves, Uzay Eğrileri)" align="right" height="140"/>

[![PyPI version](https://badge.fury.io/py/spacecurves.svg)](https://badge.fury.io/py/spacecurves/)
[![License: AGPL](https://img.shields.io/badge/License-AGPL-yellow.svg)](https://opensource.org/licenses/AGPL)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.***.svg)](https://doi.org/10.5281/zenodo.***)
[![WorkflowHub DOI](https://img.shields.io/badge/DOI-10.48546%2Fworkflowhub.datafile.***-blue)](https://doi.org/10.48546/workflowhub.datafile.***)
[![figshare DOI](https://img.shields.io/badge/DOI-10.6084/m9.figshare.***-blue)](https://doi.org/10.6084/m9.figshare.***)

[![Anaconda-Server Badge](https://anaconda.org/bilgi/spacecurves/badges/version.svg)](https://anaconda.org/bilgi/spacecurves)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/spacecurves/badges/latest_release_date.svg)](https://anaconda.org/bilgi/spacecurves)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/spacecurves/badges/platforms.svg)](https://anaconda.org/bilgi/spacecurves)
[![Anaconda-Server Badge](https://anaconda.org/bilgi/spacecurves/badges/license.svg)](https://anaconda.org/bilgi/spacecurves)

[![Open Source](https://img.shields.io/badge/Open%20Source-Open%20Source-brightgreen.svg)](https://opensource.org/)
[![Documentation Status](https://app.readthedocs.org/projects/spacecurves/badge/?0.2.0=main)](https://spacecurves.readthedocs.io/en/latest)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/10536/badge)](https://www.bestpractices.dev/projects/10536)

[![Python CI](https://github.com/WhiteSymmetry/spacecurves/actions/workflows/python_ci.yml/badge.svg?branch=main)](https://github.com/WhiteSymmetry/spacecurves/actions/workflows/python_ci.yml)
[![codecov](https://codecov.io/gh/WhiteSymmetry/spacecurves/graph/badge.svg?token=0X78S7TL0W)](https://codecov.io/gh/WhiteSymmetry/spacecurves)
[![Documentation Status](https://readthedocs.org/projects/spacecurves/badge/?version=latest)](https://spacecurves.readthedocs.io/en/latest/)
[![Binder](https://terrarium.evidencepub.io/badge_logo.svg)](https://terrarium.evidencepub.io/v2/gh/WhiteSymmetry/spacecurves/HEAD)

[![PyPI version](https://badge.fury.io/py/spacecurves.svg)](https://badge.fury.io/py/spacecurves)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![Linted with Ruff](https://img.shields.io/badge/Linted%20with-Ruff-green?logo=python&logoColor=white)](https://github.com/astral-sh/ruff)
[![Lang:Python](https://img.shields.io/badge/Lang-Python-blue?style=flat-square&logo=python)](https://python.org/)

[![PyPI Downloads](https://static.pepy.tech/badge/spacecurves)](https://pepy.tech/projects/spacecurves)
![PyPI Downloads](https://img.shields.io/pypi/dm/spacecurves?logo=pypi&label=PyPi%20downloads)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/spacecurves?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/spacecurves)

---

<p align="left">
    <table>
        <tr>
            <td style="text-align: center;">PyPI</td>
            <td style="text-align: center;">
                <a href="https://pypi.org/project/spacecurves/">
                    <img src="https://badge.fury.io/py/spacecurves.svg" alt="PyPI version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">Conda</td>
            <td style="text-align: center;">
                <a href="https://anaconda.org/bilgi/spacecurves">
                    <img src="https://anaconda.org/bilgi/spacecurves/badges/version.svg" alt="conda-forge version" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">DOI</td>
            <td style="text-align: center;">
                <a href="https://doi.org/10.5281/zenodo.***">
                    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.***.svg" alt="DOI" height="18"/>
                </a>
            </td>
        </tr>
        <tr>
            <td style="text-align: center;">License: AGPL</td>
            <td style="text-align: center;">
                <a href="https://opensource.org/licenses/AGPL">
                    <img src="https://img.shields.io/badge/License-AGPL-yellow.svg" alt="License" height="18"/>
                </a>
            </td>
        </tr>
    </table>
</p>

---

**Space Curves (spacecurves, Uzay Eğrileri)** 

```markdown
# Space Curves (spacecurves) / Uzay Eğrileri

[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)]()

> **English** | [Türkçe](#türkçe)

---

## English

### 📌 Overview

**Space Curves (spacecurves)** is a comprehensive Python module for space-filling curves, providing a complete implementation of Hilbert curves and other space-filling curves with advanced features.

### ✨ Features

- **Multi-dimensional support**: 1-5 dimensions
- **Configurable depth**: 1-10 iterations (2^p grid size)
- **4 curve types**: Hilbert, Morton
- **High performance**: LRU cache support, batch operations
- **Neighbor search**: Find neighbors in Hilbert space
- **Image compression**: Hilbert curve-based image compression
- **Dimension reduction**: High-dimensional data → 2D visualization
- **Hilbert ordering**: Spatial data organization
- **Grid system**: Hilbert-based data structure
- **Path optimization**: TSP-like route optimization
- Görselleştirme araçları (HilbertVisualizer)
- Yaklaşık kümeleme (HilbertClustering)

### 🚀 Quick Start

```python
from spacecurves import SpaceFillingCurve, CurveType

# Create a 2D Hilbert curve (16x16 grid)
curve = SpaceFillingCurve(p=4, n=2)

# Transform: distance → coordinates
point = curve[42]           # [7, 0]

# Inverse transform: coordinates → distance
distance = curve.inverse([7, 0])  # 42

# Batch transform
distances = [0, 10, 20, 30, 40]
points = curve.batch_transform(distances)

# Find neighbors
neighbors = curve.get_neighbors([5, 5], radius=2)
```

### 📦 Installation

```bash
pip install spacecurves
```

Or directly from source:

```bash
git clone https://github.com/yourusername/spacecurves.git
cd spacecurves
pip install -e .
```

### 📚 Examples

#### 1. GPS Coordinate Indexing

```python
from spacecurves import SpaceFillingCurve

curve = SpaceFillingCurve(p=8, n=2)

# City coordinates (latitude, longitude)
cities = {
    'Istanbul': [41.0082, 28.9784],
    'Ankara': [39.9334, 32.8597],
}

# Convert to Hilbert index
for name, coord in cities.items():
    x = int((coord[0] - 36) / 6 * 255)
    y = int((coord[1] - 26) / 15 * 255)
    idx = curve.inverse([x, y])
    print(f"{name}: {idx}")
```

#### 2. Image Compression

```python
import numpy as np
from spacecurves import HilbertImageCompressor

# Create test image
image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

# Compress
compressor = HilbertImageCompressor(image)
compressed = compressor.compress(keep_ratio=0.2)  # Keep 20% of pixels
```

#### 3. Dimension Reduction

```python
from spacecurves import HilbertDimensionReducer

# 10D data → 2D visualization
reducer = HilbertDimensionReducer(target_dim=2, p=6)
reduced = reducer.fit_transform(high_dim_data)
```

#### 4. Route Optimization

```python
from spacecurves import HilbertPathOptimizer

curve = SpaceFillingCurve(p=8, n=2)
optimizer = HilbertPathOptimizer(curve)

# Optimize delivery route
optimized_route = optimizer.optimize_order(delivery_points)
```

#### 5. Hilbert Grid

```python
from spacecurves import HilbertGrid

grid = HilbertGrid(curve)
grid[[10, 20]] = "Value A"
grid[[30, 40]] = "Value B"

print(grid[[10, 20]])  # Value A
```

### 🔧 API Reference

#### `SpaceFillingCurve(p, n, curve_type=CurveType.HILBERT, use_cache=True)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `p` | int | required | Iteration count (2^p grid size), 1-10 |
| `n` | int | required | Number of dimensions, 1-5 |
| `curve_type` | CurveType | HILBERT | Curve type |
| `use_cache` | bool | True | Enable caching |

**Methods:**

| Method | Description |
|--------|-------------|
| `transform(distance)` | Convert distance to coordinates |
| `inverse(point)` | Convert coordinates to distance |
| `batch_transform(distances)` | Batch conversion |
| `batch_inverse(points)` | Batch inverse conversion |
| `get_neighbors(point, radius)` | Find neighbors |
| `sample(n_points, method)` | Sample points along curve |

#### Curve Types

| Type | Description | Best For |
|------|-------------|----------|
| `HILBERT` | Classic Hilbert curve | General purpose |
| `MORTON` | Z-order curve | Fast indexing |
| `MOORE` | Moore curve (Hilbert variant) | Closed loops |
| `ALTAIR` | Hybrid curve | Speed/locality balance |

### 📊 Performance

- **Transform speed**: ~2.6M ops/sec (2D, p=4)
- **Cache hit rate**: >95% with repeated queries
- **Grid access**: ~0.01ms per operation

### 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**.

### 👤 Author

**Mehmet Keçeci**  
Email: mkececi@yaani.com

---

## Türkçe

### 📌 Genel Bakış

**Space Curves (spacecurves)**, uzay dolduran eğriler için kapsamlı bir Python modülüdür. Hilbert eğrisi ve diğer uzay dolduran eğrilerin tam bir implementasyonunu gelişmiş özelliklerle sunar.

### ✨ Özellikler

- **Çok boyutlu destek**: 1-5 boyut
- **Ayarlanabilir derinlik**: 1-16 iterasyon (2^p grid boyutu)
- **4 eğri tipi**: Hilbert, Morton, Moore, Altair
- **Yüksek performans**: LRU önbellek, toplu işlemler
- **Komşuluk arama**: Hilbert uzayında komşu bulma
- **Görüntü sıkıştırma**: Hilbert eğrisi tabanlı görüntü sıkıştırma
- **Boyut indirgeme**: Yüksek boyutlu veri → 2D görselleştirme
- **Hilbert sıralama**: Mekansal veri düzenleme
- **Grid sistemi**: Hilbert tabanlı veri yapısı
- **Rota optimizasyonu**: TSP benzeri rota optimizasyonu
- Görselleştirme araçları (HilbertVisualizer)
- Yaklaşık kümeleme (HilbertClustering)

### 🚀 Hızlı Başlangıç

```python
from spacecurves import SpaceFillingCurve, CurveType

# 2D Hilbert eğrisi oluştur (16x16 grid)
curve = SpaceFillingCurve(p=4, n=2)

# Dönüşüm: mesafe → koordinat
point = curve[42]           # [7, 0]

# Ters dönüşüm: koordinat → mesafe
distance = curve.inverse([7, 0])  # 42

# Toplu dönüşüm
distances = [0, 10, 20, 30, 40]
points = curve.batch_transform(distances)

# Komşu bulma
neighbors = curve.get_neighbors([5, 5], radius=2)
```

### 📦 Kurulum

```bash
pip install spacecurves
```

Veya doğrudan kaynaktan:

```bash
git clone https://github.com/WhiteSymmetry/spacecurves.git
cd spacecurves
pip install -e .
```

### 📚 Örnekler

#### 1. GPS Koordinat İndeksleme

```python
from spacecurves import SpaceFillingCurve

curve = SpaceFillingCurve(p=8, n=2)

# Şehir koordinatları (enlem, boylam)
sehirler = {
    'İstanbul': [41.0082, 28.9784],
    'Ankara': [39.9334, 32.8597],
}

# Hilbert indeksine dönüştür
for ad, koord in sehirler.items():
    x = int((koord[0] - 36) / 6 * 255)
    y = int((koord[1] - 26) / 15 * 255)
    idx = curve.inverse([x, y])
    print(f"{ad}: {idx}")
```

#### 2. Görüntü Sıkıştırma

```python
import numpy as np
from spacecurves import HilbertImageCompressor

# Test görüntüsü oluştur
gorsel = np.random.randint(0, 256, (256, 256), dtype=np.uint8)

# Sıkıştır
sikistirici = HilbertImageCompressor(gorsel)
sikistirilmis = sikistirici.compress(keep_ratio=0.2)  # Piksellerin %20'sini tut
```

#### 3. Boyut İndirgeme

```python
from spacecurves import HilbertDimensionReducer

# 10D veri → 2D görselleştirme
indirgeyici = HilbertDimensionReducer(target_dim=2, p=6)
indirgenmis = indirgeyici.fit_transform(yuksek_boyutlu_veri)
```

#### 4. Rota Optimizasyonu

```python
from spacecurves import HilbertPathOptimizer

curve = SpaceFillingCurve(p=8, n=2)
optimizer = HilbertPathOptimizer(curve)

# Teslimat rotasını optimize et
optimize_rota = optimizer.optimize_order(teslimat_noktalari)
```

#### 5. Hilbert Grid

```python
from spacecurves import HilbertGrid

grid = HilbertGrid(curve)
grid[[10, 20]] = "Değer A"
grid[[30, 40]] = "Değer B"

print(grid[[10, 20]])  # Değer A
```

### 🔧 API Referansı

#### `SpaceFillingCurve(p, n, curve_type=CurveType.HILBERT, use_cache=True)`

| Parametre | Tip | Varsayılan | Açıklama |
|-----------|-----|------------|----------|
| `p` | int | gerekli | İterasyon sayısı (2^p grid boyutu), 1-10 |
| `n` | int | gerekli | Boyut sayısı, 1-5 |
| `curve_type` | CurveType | HILBERT | Eğri tipi |
| `use_cache` | bool | True | Önbellek kullanımı |

**Metotlar:**

| Metot | Açıklama |
|-------|-----------|
| `transform(distance)` | Mesafeyi koordinata dönüştürür |
| `inverse(point)` | Koordinatı mesafeye dönüştürür |
| `batch_transform(distances)` | Toplu dönüşüm |
| `batch_inverse(points)` | Toplu ters dönüşüm |
| `get_neighbors(point, radius)` | Komşuları bulur |
| `sample(n_points, method)` | Eğri boyunca örnek noktalar |

#### Eğri Tipleri

| Tip | Açıklama | En İyi Kullanım Alanı |
|-----|----------|----------------------|
| `HILBERT` | Klasik Hilbert eğrisi | Genel amaçlı |
| `MORTON` | Z-order eğrisi | Hızlı indeksleme |
| `MOORE` | Moore eğrisi (Hilbert varyantı) | Kapalı döngüler |
| `ALTAIR` | Hibrid eğri | Hız/lokalite dengesi |

### 📊 Performans

- **Dönüşüm hızı**: ~2.6M işlem/saniye (2D, p=4)
- **Önbellek isabet oranı**: Tekrarlı sorgularda >%95
- **Grid erişimi**: ~0.01ms işlem başına

### 📄 Lisans

Bu proje **GNU Affero General Public License v3.0 (AGPL-3.0-or-later)** ile lisanslanmıştır.

### 👤 Yazar

**Mehmet Keçeci**  

---

## Acknowledgments / Teşekkürler

- Hilbert curve algorithm based on the work of David Hilbert (1891)
- Morton order (Z-order curve) by Guy Macdonald Morton (1966)
- Moore curve by E. H. Moore (1900)


```

---

# Pixi:

[![Pixi](https://img.shields.io/badge/Pixi-Pixi-brightgreen.svg)](https://prefix.dev/channels/bilgi)

pixi init spacecurves

cd spacecurves

pixi workspace channel add [https://prefix.dev/channels/bilgi](https://prefix.dev/channels/bilgi) --prepend

✔ Added https://prefix.dev/channels/bilgi

pixi add spacecurves

✔ Added spacecurves >=...,<1

pixi install

pixi shell

pixi run python -c "import spacecurves; print(spacecurves.__version__)"

### Çıktı: 

pixi remove spacecurves

conda install -c https://prefix.dev/channels/bilgi spacecurves

pixi run python -c "import spacecurves; print(spacecurves.__version__)"

### Çıktı: 

pixi run pip list | grep spacecurves

### spacecurves  

pixi run pip show spacecurves

Name: spacecurves

Version: 

Summary: Keçeci Numbers: Keçeci Sayıları (Keçeci Conjecture)

Home-page: https://github.com/WhiteSymmetry/spacecurves

Author: Mehmet Keçeci

Author-email: Mehmet Keçeci <...>

License: GNU AFFERO GENERAL PUBLIC LICENSE

Copyright (c) 2025-2026 Mehmet Keçeci
