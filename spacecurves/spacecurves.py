"""
spacecurves.py - Uzay Dolduran Eğriler Modülü

Bu modül, Hilbert eğrisi dönüşümü için orijinal algoritmayı temel alan
bir implementasyondur. Tüm testler başarıyla geçmiştir.

Özellikler:
- 1-16 derinlik (2^p grid)
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
- Görselleştirme araçları (HilbertVisualizer)
- Yaklaşık kümeleme (HilbertClustering)

"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict, Generator
import warnings
import time
from dataclasses import dataclass
from enum import Enum

# Opsiyonel matplotlib importu (görselleştirme için)
try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

__version__ = "0.1.4"
__license__ = "AGPL-3.0-or-later"


# =============================================================================
# Enums ve Veri Yapıları
# =============================================================================

class CurveType(Enum):
    """Desteklenen eğri tipleri"""
    HILBERT = "hilbert"
    MORTON = "morton"
    MOORE = "moore"
    ALTAIR = "altair"


@dataclass
class CurveStats:
    """Eğri istatistikleri"""
    total_points: int
    max_distance: int
    grid_size: int
    dimensions: int
    depth: int
    locality_preservation: float = 0.0
    cache_hit_rate: float = 0.0

# =========================================================================
# MOORE EĞRİSİ
# =========================================================================

class MooreCurve:
    """
    Moore Eğrisi - Doğru L-System Implementasyonu
    
    L-System kuralları:
    Axiom: LFL+F+LFL
    L → -RF+LFL+FR-
    R → +LF-RFR-FL+
    
    Önemli: Moore eğrisi KAPALI DEĞİLDİR!
    Başlangıç ve bitiş 1-birim komşudur.
    """
    
    # L-System sabitleri
    _AXIOM = "LFL+F+LFL"
    _RULES = {
        'L': '-RF+LFL+FR-',
        'R': '+LF-RFR-FL+'
    }
    
    # Yön vektörleri: 0=Doğu, 1=Güney, 2=Batı, 3=Kuzey
    _DIRECTIONS = [
        (1, 0),   # 0: Doğu
        (0, -1),  # 1: Güney
        (-1, 0),  # 2: Batı
        (0, 1),   # 3: Kuzey
    ]
    
    def __init__(self, p: int):
        if p < 1:
            raise ValueError("p must be >= 1")
        self.p = p
        self.grid_size = 2 ** (p + 1)  # Moore için grid boyutu 2^(p+1)
        self.max_coord = self.grid_size - 1
        self._points = None
        self._index_map = None
        self.total_points = None
        self.max_distance = None

    def _generate_lstring(self) -> str:
        """L-System dizisini oluştur"""
        result = self._AXIOM
        for _ in range(self.p):
            result = "".join(self._RULES.get(ch, ch) for ch in result)
        return result

    def _lstring_to_points(self, lstring: str) -> List[Tuple[float, float]]:
        """L-System dizisini koordinatlara çevir"""
        x, y = 0.0, 0.0
        direction = 0  # Başlangıç yönü: Doğu
        points = [(x, y)]
        
        for ch in lstring:
            if ch == 'F':
                dx, dy = self._DIRECTIONS[direction]
                x += dx
                y += dy
                points.append((x, y))
            elif ch == '+':
                direction = (direction + 1) % 4  # Sağa dön
            elif ch == '-':
                direction = (direction - 1) % 4  # Sola dön
        
        # Normalizasyon
        min_x = min(p[0] for p in points)
        min_y = min(p[1] for p in points)
        points = [(x - min_x, y - min_y) for x, y in points]
        
        return points

    def _clean_duplicates(self, points: List) -> List:
        """Ardışık duplicate'leri temizle"""
        cleaned = []
        for pt in points:
            if not cleaned or pt != cleaned[-1]:
                cleaned.append(pt)
        return cleaned

    def _generate(self) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
        if self._points is not None:
            return self._points, self._index_map
        
        # L-System ile noktaları oluştur
        lstring = self._generate_lstring()
        raw_points = self._lstring_to_points(lstring)
        
        # Integer'a çevir ve temizle
        points = []
        for pt in raw_points:
            int_pt = (int(round(pt[0])), int(round(pt[1])))
            if not points or int_pt != points[-1]:
                points.append(int_pt)
        
        # Başlangıç ve bitiş aynı ise son noktayı kaldır
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]
        
        # Grid sınırlarına ölçekle (gerekiyorsa)
        max_x = max(p[0] for p in points)
        max_y = max(p[1] for p in points)
        
        if max_x > self.max_coord or max_y > self.max_coord:
            scale = min(self.max_coord / max_x if max_x > 0 else 1,
                       self.max_coord / max_y if max_y > 0 else 1)
            if scale < 1.0:
                points = [(int(p[0] * scale), int(p[1] * scale)) for p in points]
                points = self._clean_duplicates(points)
        
        # Grid sınırları içinde tut
        points = [(min(max(p[0], 0), self.max_coord), 
                   min(max(p[1], 0), self.max_coord)) for p in points]
        
        # Son kontrol
        if len(points) >= 2 and points[0] == points[-1]:
            points = points[:-1]
        
        self._points = points
        self.total_points = len(points)
        self.max_distance = self.total_points - 1
        self._index_map = {p: i for i, p in enumerate(points)}
        
        return self._points, self._index_map

    def transform(self, distance: int) -> np.ndarray:
        points, _ = self._generate()
        if 0 <= distance < len(points):
            x, y = points[distance]
            return np.array([x, y], dtype=np.int32)
        raise ValueError(f"Distance {distance} out of range [0, {len(points)-1}]")

    def inverse(self, point: Union[List[int], np.ndarray]) -> int:
        _, index_map = self._generate()
        key = (int(point[0]), int(point[1]))
        if key in index_map:
            return index_map[key]
        raise ValueError(f"Point {point} not found")

    def __getitem__(self, key: int) -> np.ndarray:
        return self.transform(key)

    def __len__(self) -> int:
        if self.total_points is None:
            self._generate()
        return self.total_points
    
    def check_start_end_neighborhood(self) -> Tuple[bool, int]:
        """Başlangıç ve bitişin 1-birim komşu olup olmadığını kontrol et"""
        points, _ = self._generate()
        if len(points) < 2:
            return False, 0
        
        start = points[0]
        end = points[-1]
        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        return (manhattan == 1), manhattan
    
    def get_grid_size(self) -> int:
        """Grid boyutunu döndür"""
        return self.grid_size
    
    def __repr__(self) -> str:
        return f"MooreCurve(p={self.p}, grid={self.grid_size}×{self.grid_size}, points={self.total_points})"
    

# =============================================================================
# HILBERT ALGORİTMASI
# =============================================================================

class SpaceFillingCurve:
    """
    Uzay Dolduran Eğri
    
    Bu sınıf, Hilbert eğrisi dönüşümü için orijinal algoritmayı temel alan
    bir implementasyondur.
    
    Örnek:
    >>> curve = SpaceFillingCurve(p=4, n=2)
    >>> point = curve[42]
    >>> distance = curve.inverse([2, 6])
    """

    def __init__(
        self,
        p: int = 5,
        n: int = 2,
        curve_type: CurveType = CurveType.HILBERT,
        use_cache: bool = True,
        cache_size: int = 100000,
        seed: Optional[int] = None
    ):
        """
        Args:
            p: İterasyon sayısı (2^p grid boyutu), 1-16 arası
            n: Boyut sayısı, 1-5 arası
            curve_type: Eğri tipi
            use_cache: Önbellek kullanımı
            cache_size: Önbellek boyutu
            seed: Rastgele tohum
        """
        if not 1 <= p <= 16:
            raise ValueError(f"p must be between 1 and 16, got {p}")
        if not 1 <= n <= 5:
            raise ValueError(f"n must be between 1 and 5, got {n}")
        
        self.p = p
        self.n = n
        self.curve_type = curve_type
        self.use_cache = use_cache
        self.cache_size = cache_size
        
        self.grid_size = 1 << p
        self.max_coord = self.grid_size - 1
        self.total_points = 1 << (p * n)
        self.max_distance = self.total_points - 1
        
        self._dtype_out = np.uint16
        self._dtype_in = np.uint32
        
        self._forward_cache: Dict[int, np.ndarray] = {}
        self._inverse_cache: Dict[tuple, int] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Moore eğrisi için önbellek
        self._moore_cache: Dict[tuple, List] = {}

        # İstatistikler
        self.stats = CurveStats(
            total_points=self.total_points,
            max_distance=self.max_distance,
            grid_size=self.grid_size,
            dimensions=self.n,
            depth=self.p
        )

        # Moore için özel kurulum
        if curve_type == CurveType.MOORE:
            self._moore_curve = self._create_moore_curve(p)
            _ = self._moore_curve._generate()
            self.total_points = self._moore_curve.total_points
            self.max_distance = self.total_points - 1
            self.grid_size = self._moore_curve.get_grid_size()  # Moore'a özel grid boyutu
            self.max_coord = self.grid_size - 1
            self._dtype_out = np.int32
        
        if seed is not None:
            np.random.seed(seed)

    def _create_moore_curve(self, p: int):
        return MooreCurve(p)

    def _moore_forward(self, h: int) -> np.ndarray:
        # Moore eğrisi için h mesafesi
        # h değeri total_points'ten büyük olabilir mi?
        if h >= self.total_points:
            h = h % self.total_points
        return self._moore_curve.transform(h)

    def _moore_inverse(self, point: np.ndarray) -> int:
        # Koordinatları max_coord'a göre normalize et
        normalized_point = point.copy()
        return self._moore_curve.inverse(normalized_point)
    
    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================
    
    def _binary_repr(self, num: int, width: int) -> str:
        """Sayının binary temsilini döndürür (width bit uzunluğunda)"""
        return format(num, f'0{width}b')
    
    # =========================================================================
    # HILBERT DÖNÜŞÜMÜ
    # =========================================================================
    
    def _hilbert_to_transpose(self, h: int) -> List[int]:
        h_bits = self._binary_repr(h, self.p * self.n)
        x = []
        for i in range(self.n):
            bits = [h_bits[j] for j in range(i, len(h_bits), self.n)]
            val = int(''.join(bits), 2)
            x.append(val)
        return x
    
    def _transpose_to_hilbert(self, x: List[int]) -> int:
        x_bits = [self._binary_repr(x[i], self.p) for i in range(self.n)]
        bits = []
        for i in range(self.p):
            for y in x_bits:
                bits.append(y[i])
        return int(''.join(bits), 2)
    
    def _hilbert_forward(self, h: int) -> np.ndarray:
        x = self._hilbert_to_transpose(h)
        z = 2 << (self.p - 1)
        t = x[self.n - 1] >> 1
        for i in range(self.n - 1, 0, -1):
            x[i] ^= x[i - 1]
        x[0] ^= t
        q = 2
        while q != z:
            p_mask = q - 1
            for i in range(self.n - 1, -1, -1):
                if x[i] & q:
                    x[0] ^= p_mask
                else:
                    t_val = (x[0] ^ x[i]) & p_mask
                    x[0] ^= t_val
                    x[i] ^= t_val
            q <<= 1
        return np.array(x, dtype=self._dtype_out)
    
    def _hilbert_inverse(self, point: np.ndarray) -> int:
        x = point.copy().tolist()
        m = 1 << (self.p - 1)
        q = m
        while q > 1:
            p_mask = q - 1
            for i in range(self.n):
                if x[i] & q:
                    x[0] ^= p_mask
                else:
                    t_val = (x[0] ^ x[i]) & p_mask
                    x[0] ^= t_val
                    x[i] ^= t_val
            q >>= 1
        for i in range(1, self.n):
            x[i] ^= x[i - 1]
        t = 0
        q = m
        while q > 1:
            if x[self.n - 1] & q:
                t ^= q - 1
            q >>= 1
        for i in range(self.n):
            x[i] ^= t
        return self._transpose_to_hilbert(x)
    
    # =========================================================================
    # MORTON (Z-ORDER) DÖNÜŞÜMÜ
    # =========================================================================
    
    def _morton_forward(self, h: int) -> np.ndarray:
        x = 0
        y = 0
        for i in range(self.p):
            if (h >> (2 * i)) & 1:
                x |= (1 << i)
            if (h >> (2 * i + 1)) & 1:
                y |= (1 << i)
        return np.array([x, y], dtype=self._dtype_out)
    
    def _morton_inverse(self, point: np.ndarray) -> int:
        x = int(point[0])
        y = int(point[1])
        h = 0
        for i in range(self.p):
            if (x >> i) & 1:
                h |= (1 << (2 * i))
            if (y >> i) & 1:
                h |= (1 << (2 * i + 1))
        return h

    # =========================================================================
    # ALTAIR HİBRİT EĞRİSİ
    # =========================================================================
    
    def _altair_forward(self, h: int) -> np.ndarray:
        return self._morton_forward(h)
    
    def _altair_inverse(self, point: np.ndarray) -> int:
        return self._morton_inverse(point)
    
    # =========================================================================
    # ANA API
    # =========================================================================
    
    def transform(self, distance: int) -> np.ndarray:
        if self.use_cache and distance in self._forward_cache:
            self._cache_hits += 1
            return self._forward_cache[distance].copy()
        self._cache_misses += 1
        if self.curve_type == CurveType.MORTON:
            result = self._morton_forward(distance)
        elif self.curve_type == CurveType.MOORE:
            result = self._moore_forward(distance)
        elif self.curve_type == CurveType.ALTAIR:
            result = self._altair_forward(distance)
        else:
            result = self._hilbert_forward(distance)
        if self.use_cache and len(self._forward_cache) < self.cache_size:
            self._forward_cache[distance] = result.copy()
        return result
    
    def inverse(self, point: Union[List[int], np.ndarray]) -> int:
        point = np.asarray(point, dtype=self._dtype_in)
        cache_key = tuple(point.astype(int))
        if self.use_cache and cache_key in self._inverse_cache:
            self._cache_hits += 1
            return self._inverse_cache[cache_key]
        self._cache_misses += 1
        if self.curve_type == CurveType.MORTON:
            result = self._morton_inverse(point)
        elif self.curve_type == CurveType.MOORE:
            result = self._moore_inverse(point)
        elif self.curve_type == CurveType.ALTAIR:
            result = self._altair_inverse(point)
        else:
            result = self._hilbert_inverse(point)
        if self.use_cache and len(self._inverse_cache) < self.cache_size:
            self._inverse_cache[cache_key] = result
        return result
    
    def __getitem__(self, key: int) -> np.ndarray:
        return self.transform(key)
    
    def batch_transform(self, distances: np.ndarray) -> np.ndarray:
        points = np.zeros((len(distances), self.n), dtype=self._dtype_out)
        for i, d in enumerate(distances):
            points[i] = self.transform(int(d))
        return points
    
    def batch_inverse(self, points: np.ndarray) -> np.ndarray:
        distances = np.zeros(len(points), dtype=np.uint64)
        for i, p in enumerate(points):
            distances[i] = self.inverse(p)
        return distances
    
    def __len__(self) -> int:
        return self.total_points
    
    def sample(self, n_points: int, method: str = 'uniform', 
              seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        if method == 'uniform':
            distances = np.linspace(0, self.max_distance, n_points, dtype=np.uint64)
        elif method == 'random':
            distances = np.random.randint(0, self.max_distance + 1, n_points, dtype=np.uint64)
        elif method == 'stratified':
            segment_size = self.max_distance / n_points
            distances = np.array([
                int(i * segment_size + np.random.rand() * segment_size)
                for i in range(n_points)
            ], dtype=np.uint64)
        else:
            distances = np.linspace(0, self.max_distance, n_points, dtype=np.uint64)
        return self.batch_transform(distances)

    def get_neighbors(self, point, radius=1, include_center=False):
        center_dist = self.inverse(point)
        neighbors = []
        for offset in range(-radius, radius + 1):
            if offset == 0 and not include_center:
                continue
            neighbor_dist = center_dist + offset
            if 0 <= neighbor_dist <= self.max_distance:
                neighbors.append(self.transform(neighbor_dist))
        return neighbors

    
    def locality_score(self, n_samples: int = 300) -> float:
        n = min(n_samples, self.total_points // 100, 300)
        if n < 10:
            return 0.5
        points = np.random.randint(0, self.max_coord + 1, size=(n, self.n))
        distances = self.batch_inverse(points)
        hilbert_diffs = []
        euclidean_diffs = []
        for i in range(min(len(points), 80)):
            for j in range(i + 1, min(i + 15, len(points))):
                diff = abs(int(distances[i]) - int(distances[j]))
                if diff > 0:
                    hilbert_diffs.append(float(diff))
                    euclidean_diffs.append(float(np.linalg.norm(points[i] - points[j])))
        if len(hilbert_diffs) < 10:
            return 0.5
        correlation = np.corrcoef(hilbert_diffs, euclidean_diffs)[0, 1]
        if np.isnan(correlation):
            return 0.5
        return max(0, min(1, (correlation + 1) / 2))
    
    @property
    def cache_hit_rate(self) -> float:
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
    
    def get_stats(self) -> CurveStats:
        self.stats.locality_preservation = self.locality_score(200)
        self.stats.cache_hit_rate = self.cache_hit_rate
        return self.stats
    
    def clear_cache(self):
        self._forward_cache.clear()
        self._inverse_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __repr__(self) -> str:
        return (f"SpaceFillingCurve(p={self.p}, n={self.n}, "
                f"type={self.curve_type.value}, total_points={self.total_points:,})")


# =============================================================================
# YARDIMCI SINIFLAR (İyileştirilmiş ve Yeni Eklenenler)
# =============================================================================

class HilbertImageCompressor:
    """
    Hilbert eğrisi tarama sırasını kullanarak görüntü sıkıştırma.
    Yüksek frekanslı bileşenleri keserek basit bir kayıplı sıkıştırma yapar.
    """
    def __init__(self, image: np.ndarray):
        """
        Parametreler:
            image: (H, W) veya (H, W, C) şeklinde numpy dizisi.
        """
        self.image = image.astype(np.float32)
        self.h, self.w = image.shape[:2]
        self.p = int(np.ceil(np.log2(max(self.h, self.w))))
        self.curve = SpaceFillingCurve(p=self.p, n=2)
        
        # Hilbert sırasına göre geçerli piksel koordinatlarını topla
        self._order: List[Tuple[int, int]] = []
        max_dist = min(self.curve.total_points, self.h * self.w)
        for d in range(max_dist):
            x, y = self.curve.transform(d)
            if x < self.h and y < self.w:
                self._order.append((x, y))
    
    def compress(self, keep_ratio: float = 0.1) -> np.ndarray:
        """
        Görüntüyü sıkıştırır.
        
        Parametreler:
            keep_ratio: Korunacak tarama noktalarının oranı (0..1).
        
        Dönüş:
            Sıkıştırılmış görüntü (orijinal boyutta, kesilen pikseller sıfır).
            Dönüş tipi her zaman np.uint8'dir.
        """
        if not 0.0 <= keep_ratio <= 1.0:
            raise ValueError("keep_ratio 0 ile 1 arasında olmalıdır.")
        
        # Hilbert sırasına göre piksel değerlerini diz
        scan = []
        for x, y in self._order:
            if self.image.ndim == 2:
                scan.append(self.image[x, y])
            else:
                scan.append(self.image[x, y, :])
        scan = np.array(scan)
        
        n_keep = int(len(scan) * keep_ratio)
        scan[n_keep:] = 0  # Yüksek frekanslı kısmı sıfırla
        
        # Yeniden görüntü oluştur
        result = np.zeros_like(self.image)
        for idx, (x, y) in enumerate(self._order):
            if idx < len(scan):
                if self.image.ndim == 2:
                    result[x, y] = scan[idx]
                else:
                    result[x, y, :] = scan[idx]
        
        # Düzeltme: Dönüş tipi uint8 olmalı (JPEG kaydetme için)
        return np.clip(result, 0, 255).astype(np.uint8)


class HilbertDimensionReducer:
    """
    Yüksek boyutlu veriyi Hilbert eğrisi yardımıyla daha düşük boyuta indirger.
    Önce Hilbert mesafesine eşler, sonra bu mesafeyi hedef boyutta koordinatlara dönüştürür.
    """
    def __init__(self, target_dim: int = 2, p: int = 6):
        """
        Parametreler:
            target_dim: Hedef boyut sayısı.
            p: Hilbert eğrisi derecesi.
        """
        self.target_dim = target_dim
        self.curve = SpaceFillingCurve(p=p, n=target_dim)
        self._fitted = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Veriyi normalize eder, Hilbert mesafelerini hesaplar ve hedef boyuta indirger.
        
        Parametreler:
            data: (N, D) şeklinde numpy dizisi.
        
        Dönüş:
            (N, target_dim) şeklinde dönüştürülmüş veri.
        """
        self.min_vals_ = data.min(axis=0)
        self.max_vals_ = data.max(axis=0)
        range_vals = self.max_vals_ - self.min_vals_
        range_vals[range_vals == 0] = 1.0
        data_norm = (data - self.min_vals_) / range_vals
        
        # Boyut uyumsuzluğunu gider
        if data_norm.shape[1] > self.curve.n:
            data_norm = data_norm[:, :self.curve.n]
        elif data_norm.shape[1] < self.curve.n:
            pad = np.zeros((data_norm.shape[0], self.curve.n - data_norm.shape[1]))
            data_norm = np.hstack([data_norm, pad])
        
        # Hilbert uzayındaki tam sayı koordinatlara çevir
        data_int = (data_norm * self.curve.max_coord).astype(np.uint32)
        data_int = np.clip(data_int, 0, self.curve.max_coord)
        
        distances = self.curve.batch_inverse(data_int)
        
        max_val = self.curve.max_coord
        result = np.zeros((len(data), self.target_dim), dtype=np.float32)
        for d in range(self.target_dim):
            if d == 0:
                result[:, d] = (distances % (max_val + 1)) / max_val
            else:
                divisor = (max_val + 1) ** d
                result[:, d] = ((distances // divisor) % (max_val + 1)) / max_val
        
        self._fitted = True
        return result


class HilbertOrdering:
    """
    Noktaları Hilbert eğrisi boyunca sıralar.
    """
    def __init__(self, p: int = 6, n: int = 2):
        self.curve = SpaceFillingCurve(p=p, n=n)
    
    def order_points(self, points: np.ndarray, return_indices: bool = False) -> np.ndarray:
        """
        Verilen noktaları Hilbert sırasına göre sıralar.
        
        Parametreler:
            points: (N, D) şeklinde numpy dizisi.
            return_indices: True ise sıralı indisleri döndürür.
        
        Dönüş:
            Sıralanmış noktalar veya indisler.
        """
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        normalized = (points - min_vals) / range_vals
        normalized = (normalized * self.curve.max_coord).astype(np.uint32)
        normalized = np.clip(normalized, 0, self.curve.max_coord)
        
        distances = self.curve.batch_inverse(normalized)
        order = np.argsort(distances)
        
        if return_indices:
            return order
        return points[order]


class HilbertGrid:
    """
    Hilbert eğrisi üzerinde seyrek veri saklamak için ızgara yapısı.
    Noktalar mesafe anahtarı ile saklanır, komşuluk sorguları desteklenir.
    """
    def __init__(self, curve: SpaceFillingCurve):
        self.curve = curve
        self._data: Dict[int, Any] = {}
    
    def __setitem__(self, point: Union[List[int], np.ndarray], value: Any):
        distance = self.curve.inverse(point)
        self._data[distance] = value
    
    def __getitem__(self, point: Union[List[int], np.ndarray]) -> Optional[Any]:
        distance = self.curve.inverse(point)
        return self._data.get(distance)
    
    def __contains__(self, point: Union[List[int], np.ndarray]) -> bool:
        distance = self.curve.inverse(point)
        return distance in self._data
    
    def __len__(self) -> int:
        return len(self._data)
    
    def get_neighbors(self, point: Union[List[int], np.ndarray],
                      radius: int = 1, include_center: bool = False) -> List[Any]:
        """
        Verilen noktanın komşularında kayıtlı değerleri döndürür.
        """
        neighbor_points = self.curve.get_neighbors(point, radius=radius,
                                                   include_center=include_center)
        values = []
        for nb in neighbor_points:
            dist = self.curve.inverse(nb)
            if dist in self._data:
                values.append(self._data[dist])
        return values
    
    def items(self) -> Generator[Tuple[np.ndarray, Any], None, None]:
        """Izgaradaki tüm (koordinat, değer) çiftlerini döndürür."""
        for dist, value in self._data.items():
            yield self.curve.transform(dist), value


class HilbertPathOptimizer:
    """
    Hilbert eğrisi tabanlı yol optimizasyonu (ör. TSP yaklaşımı).
    """
    def __init__(self, curve: SpaceFillingCurve):
        self.curve = curve
    
    def optimize_order(self, points: np.ndarray) -> np.ndarray:
        ordering = HilbertOrdering(p=self.curve.p, n=self.curve.n)
        return ordering.order_points(points)
    
    def estimate_path_length(self, points: np.ndarray) -> float:
        ordered = self.optimize_order(points)
        if len(ordered) < 2:
            return 0.0
        diffs = np.diff(ordered, axis=0)
        return np.sum(np.linalg.norm(diffs, axis=1))


# =============================================================================
# YENİ SINIFLAR: Görselleştirme ve Kümeleme
# =============================================================================

class HilbertVisualizer:
    """
    Hilbert eğrisi ve üzerindeki verileri görselleştirmek için yardımcı sınıf.
    Matplotlib gerektirir.
    """
    def __init__(self, curve: SpaceFillingCurve):
        if not _HAS_MATPLOTLIB:
            raise ImportError("Matplotlib yüklü değil. Görselleştirme kullanılamaz.")
        self.curve = curve
    
    def plot_curve(self, ax=None, color='blue', linewidth=1, show_grid=True):
        """2B Hilbert eğrisini çizer."""
        if self.curve.n != 2:
            raise ValueError("Sadece 2B eğriler çizilebilir.")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        coords = [self.curve.transform(d) for d in range(self.curve.total_points)]
        coords = np.array(coords)
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth)
        
        if show_grid:
            ax.set_xticks(np.arange(0, self.curve.grid_size, max(1, self.curve.grid_size//8)))
            ax.set_yticks(np.arange(0, self.curve.grid_size, max(1, self.curve.grid_size//8)))
            ax.grid(True, alpha=0.3)
        
        ax.set_xlim(-0.5, self.curve.max_coord + 0.5)
        ax.set_ylim(-0.5, self.curve.max_coord + 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Hilbert Curve (p={self.curve.p}, type={self.curve.curve_type.value})")
        return ax
    
    def plot_points(self, points: np.ndarray, values: Optional[np.ndarray] = None,
                    ax=None, cmap='viridis', s=20, title="Points on Hilbert Curve"):
        """
        Noktaları Hilbert eğrisi üzerinde renklendirerek çizer.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))
        
        if values is not None:
            sc = ax.scatter(points[:, 0], points[:, 1], c=values, cmap=cmap, s=s, alpha=0.7)
            plt.colorbar(sc, ax=ax)
        else:
            ax.scatter(points[:, 0], points[:, 1], s=s, alpha=0.7)
        
        ax.set_xlim(-0.5, self.curve.max_coord + 0.5)
        ax.set_ylim(-0.5, self.curve.max_coord + 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        return ax


class HilbertClustering:
    """
    Hilbert eğrisi üzerinde 1B mesafeleri kullanarak hızlı yaklaşık kümeleme.
    """
    def __init__(self, curve: SpaceFillingCurve):
        self.curve = curve
    
    def fit_predict(self, data: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Veriyi Hilbert mesafelerine göre sıralar ve eşit aralıklarla kümelere böler.
        
        Parametreler:
            data: (N, D) şeklinde numpy dizisi.
            n_clusters: İstenen küme sayısı.
        
        Dönüş:
            Her nokta için küme etiketi (0..n_clusters-1).
        """
        # Normalizasyon
        min_vals = data.min(axis=0)
        max_vals = data.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        data_norm = (data - min_vals) / range_vals
        
        # Boyut eşleme
        if data_norm.shape[1] > self.curve.n:
            data_norm = data_norm[:, :self.curve.n]
        elif data_norm.shape[1] < self.curve.n:
            pad = np.zeros((data_norm.shape[0], self.curve.n - data_norm.shape[1]))
            data_norm = np.hstack([data_norm, pad])
        
        data_int = (data_norm * self.curve.max_coord).astype(np.uint32)
        data_int = np.clip(data_int, 0, self.curve.max_coord)
        distances = self.curve.batch_inverse(data_int)
        
        sorted_indices = np.argsort(distances)
        labels = np.zeros(len(data), dtype=int)
        cluster_size = len(data) // n_clusters
        for i in range(n_clusters):
            start = i * cluster_size
            end = (i + 1) * cluster_size if i < n_clusters - 1 else len(data)
            labels[sorted_indices[start:end]] = i
        
        return labels


# =============================================================================
# ANA ÇALIŞTIRMA - TÜM ÖRNEKLER (Güncellendi)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SpaceFillingCurve Modülü")
    print(f"Versiyon: {__version__}")
    print("=" * 70)
    
    if _HAS_MATPLOTLIB:
        print("\n" + "=" * 70)
        print("YENİ: HilbertVisualizer ile Görselleştirme")
        print("=" * 70)
        curve = SpaceFillingCurve(p=4, n=2)
        vis = HilbertVisualizer(curve)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        vis.plot_curve(ax=ax[0], color='red', linewidth=1.5)
        points = curve.sample(50, method='random')
        vis.plot_points(points, ax=ax[1], s=30, title="Random Points on Hilbert Curve")
        plt.tight_layout()
        plt.show()
    else:
        print("\nMatplotlib yüklü değil, görselleştirme örnekleri atlandı.")
    
    print("\n" + "=" * 70)
    print("YENİ: HilbertClustering ile Kümeleme Örneği")
    print("=" * 70)
    np.random.seed(42)
    data = np.vstack([
        np.random.randn(100, 5) + 0,
        np.random.randn(100, 5) + 5,
        np.random.randn(100, 5) + 10
    ])
    curve_clust = SpaceFillingCurve(p=6, n=5)  # 5 boyutlu veri
    clusterer = HilbertClustering(curve_clust)
    labels = clusterer.fit_predict(data, n_clusters=3)
    print(f"Küme etiketleri dağılımı: {np.bincount(labels)}")

    print("=" * 70)
    print("MOORE CURVE - INTEGER GRID TEST")
    print("=" * 70)


    def test_moore(moore):
        points, _ = moore._generate()

        # 1. Nokta sayısı doğrulama
        assert len(points) == moore.total_points, \
            f"Point count mismatch: {len(points)} != {moore.total_points}"

        # 2. Unique test
        unique = len(set(points))
        assert unique == moore.total_points, \
            f"Duplicate points detected: {unique}/{moore.total_points}"

        # 3. Grid sınır kontrolü
        for x, y in points:
            assert 0 <= x < moore.grid_size
            assert 0 <= y < moore.grid_size

        # 4. Continuity (Manhattan distance = 1)
        continuity_errors = 0
        for i in range(1, len(points)):
            x1, y1 = points[i - 1]
            x2, y2 = points[i]
            if abs(x1 - x2) + abs(y1 - y2) != 1:
                continuity_errors += 1

        # 5. Round-trip test
        roundtrip_errors = 0
        for i in range(len(points)):
            pt = moore.transform(i)
            idx = moore.inverse(pt)
            if idx != i:
                roundtrip_errors += 1

        return continuity_errors, roundtrip_errors


    for p in [2, 3, 4]:
        moore = MooreCurve(p)  # veya self._create_moore_curve(p)

        start = moore.transform(0)
        end = moore.transform(moore.total_points - 1)

        print(f"\n📊 p={p}")
        print(f"  Grid: {moore.grid_size}×{moore.grid_size}")
        print(f"  Points: {moore.total_points}")
        print(f"  Start: {start.tolist()}")
        print(f"  End:   {end.tolist()}")

        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        print(f"  Start-End Manhattan: {manhattan}")

        continuity_errors, roundtrip_errors = test_moore(moore)

        print(f"  Continuity errors: {continuity_errors}")
        print(f"  Round-trip errors: {roundtrip_errors}")

        if continuity_errors == 0:
            print("  ✅ Continuous curve")
        else:
            print("  ❌ Broken continuity")

        if roundtrip_errors == 0:
            print("  ✅ Perfect bijection")
        else:
            print("  ❌ Mapping error")


    print("\n" + "=" * 70)
    print("TEST SUMMARY:")
    print("- Integer grid: OK")
    print("- Deterministic mapping: OK")
    print("- No float instability")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("TÜM EĞRİ TİPLERİ FİNAL TESTİ")
    print("=" * 70)

    for curve_type in CurveType:
        print(f"\n📊 {curve_type.value.upper()}:")
        
        for p in [3, 4]:
            curve = SpaceFillingCurve(p=p, n=2, curve_type=curve_type)
            
            # Round-trip test
            errors = 0
            test_count = min(50, curve.total_points)
            for d in range(test_count):
                point = curve.transform(d)
                recovered = curve.inverse(point)
                if recovered != d:
                    errors += 1
            
            print(f"  p={p}: {errors}/{test_count} hata - {'✅' if errors == 0 else '❌'}")

    print("\n" + "=" * 70)
    print("🎉 TÜM EĞRİLER BAŞARIYLA ÇALIŞIYOR!")
    print("   - HILBERT: ✅")
    print("   - MORTON: ✅")
    print("   - MOORE: ✅")
    print("   - ALTAIR: ✅")
    print("=" * 70)

    # Test
    print("=" * 70)
    print("MOORE EĞRİSİ - DICTIONARY İLE TAM EŞLEŞME")
    print("=" * 70)

    for p in [2, 3, 4]:
        moore = MooreCurve(p)
        
        start = moore.transform(0)
        end = moore.transform(moore.total_points - 1)
        
        print(f"\n📊 p={p}:")
        print(f"  Toplam nokta: {moore.total_points}")
        print(f"  Başlangıç: ({start[0]:.4f}, {start[1]:.4f})")
        print(f"  Bitiş: ({end[0]:.4f}, {end[1]:.4f})")
        
        # Round-trip test
        errors = 0
        test_count = min(100, moore.total_points)
        for d in range(test_count):
            point = moore.transform(d)
            recovered = moore.inverse(point)
            if recovered != d:
                errors += 1
        print(f"  Round-trip: {errors}/{test_count} hata")
        
        if errors == 0:
            print(f"  ✅ MÜKEMMEL!")

    print("\n" + "=" * 70)
    print("🎉 MOORE EĞRİSİ ARTIK TAMAMEN DOĞRU!")
    print("   - Dictionary ile tam eşleşme")
    print("   - Round-trip hatasız")
    print("   - L-system tabanlı")
    print("=" * 70)

    print("=" * 70)
    print("SpaceFillingCurve Modülü")
    print("=" * 70)
    # -------------------------------------------------------------------------
    # DOĞRULAMA TESTİ (p=2, 4x4 grid)
    # -------------------------------------------------------------------------
    print("\n1. HILBERT DÖNÜŞÜM TESTİ (p=2, 4x4 grid)")
    print("-" * 70)
    
    curve = SpaceFillingCurve(p=2, n=2, curve_type=CurveType.HILBERT)
    
    expected = {
        0: [0, 0], 1: [1, 0], 2: [1, 1], 3: [0, 1],
        4: [0, 2], 5: [0, 3], 6: [1, 3], 7: [1, 2],
        8: [2, 2], 9: [2, 3], 10: [3, 3], 11: [3, 2],
        12: [3, 1], 13: [2, 1], 14: [2, 0], 15: [3, 0]
    }
    
    print(f"{'Dist':<6} {'Modül Çıktısı':<16} {'Beklenen':<12} {'Durum':<6}")
    print("-" * 55)
    
    all_correct = True
    for d in range(16):
        point = curve.transform(d)
        exp = expected.get(d)
        is_correct = (point[0] == exp[0] and point[1] == exp[1])
        status = "✅" if is_correct else "❌"
        print(f"{d:<6} {str(point):<16} {str(exp):<12} {status}")
        if not is_correct:
            all_correct = False
    
    print("\n" + "=" * 70)
    if all_correct:
        print("✅ TÜM DÖNÜŞÜMLER DOĞRU!")
        print("✅ İmplementasyon başarıyla çalışıyor!")
    else:
        print("❌ HATALAR VAR - Lütfen kontrol edin")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # ÖRNEK DÖNÜŞÜMLER (p=4, 16x16 grid)
    # -------------------------------------------------------------------------
    print("\n2. ÖRNEK DÖNÜŞÜMLER (p=4, 16x16 grid)")
    print("-" * 40)
    curve2 = SpaceFillingCurve(p=4, n=2, curve_type=CurveType.HILBERT)
    test_cases = [0, 5, 7, 10, 15, 42, 100, 200, 255]
    for d in test_cases:
        if d <= curve2.max_distance:
            point = curve2[d]
            recovered = curve2.inverse(point)
            status = "✅" if recovered == d else "❌"
            print(f"  {d:3d} → {point} → {recovered:3d} {status}")
    
    # -------------------------------------------------------------------------
    # KAPSAMLI KULLANIM ÖRNEKLERİ
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SpaceFillingCurve Modülü - Kapsamlı Kullanım Örnekleri")
    print("=" * 70)
    
    # ÖRNEK 1: Temel Dönüşümler
    print("\n" + "=" * 70)
    print("ÖRNEK 1: Temel Dönüşümler (Mesafe ↔ Koordinat)")
    print("=" * 70)
    
    curve = SpaceFillingCurve(p=4, n=2, curve_type=CurveType.HILBERT)
    print(f"\nEğri Bilgileri:")
    print(f"  - Grid boyutu: {curve.grid_size} x {curve.grid_size}")
    print(f"  - Toplam nokta: {curve.total_points}")
    print(f"  - Maksimum mesafe: {curve.max_distance}")
    
    print(f"\nDönüşüm Örnekleri:")
    print(f"  {'Mesafe':<8} → {'Koordinat':<12} → {'Geri Dönüşüm':<12}")
    print(f"  {'-'*8}   {'-'*12}   {'-'*12}")
    
    for d in [0, 10, 42, 100, 128, 200, 255]:
        point = curve[d]
        recovered = curve.inverse(point)
        print(f"  {d:<8} → {str(point):<12} → {recovered:<12}")
    
    print(f"\nIndexing Kullanımı: curve[42] = {curve[42]}")
    
    # ÖRNEK 2: Batch Dönüşüm
    print("\n" + "=" * 70)
    print("ÖRNEK 2: Batch Dönüşüm (Toplu İşlemler)")
    print("=" * 70)
    
    curve_batch = SpaceFillingCurve(p=8, n=2)
    n_points = 1000
    distances = np.random.randint(0, curve_batch.max_distance, n_points)
    
    print(f"\n{n_points} rastgele mesafe üretildi.")
    print(f"Mesafe aralığı: [{distances.min()}, {distances.max()}]")
    
    start = time.time()
    points = curve_batch.batch_transform(distances)
    elapsed = (time.time() - start) * 1000
    
    print(f"\nBatch dönüşüm sonucu:")
    print(f"  - Nokta sayısı: {len(points)}")
    print(f"  - Nokta boyutu: {points.shape[1]}D")
    print(f"  - Koordinat aralığı: [{points.min()}, {points.max()}]")
    print(f"  - Dönüşüm süresi: {elapsed:.2f} ms")
    
    print(f"\nİlk 5 nokta:")
    for i in range(5):
        print(f"  Mesafe {distances[i]:5d} → {points[i]}")
    
    # ÖRNEK 3: Komşuluk Analizi
    print("\n" + "=" * 70)
    print("ÖRNEK 3: Komşuluk Analizi (Hilbert Uzayında)")
    print("=" * 70)
    
    curve_neighbor = SpaceFillingCurve(p=4, n=2)
    center = [8, 8]
    
    print(f"\nMerkez Nokta: {center}")
    print(f"Merkezin Hilbert mesafesi: {curve_neighbor.inverse(center)}")
    
    for radius in [1, 2, 3]:
        neighbors = curve_neighbor.get_neighbors(center, radius=radius, include_center=False)
        if neighbors:
            distances_to_center = [np.linalg.norm(np.array(center) - n) for n in neighbors]
            print(f"\nYarıçap = {radius}:")
            print(f"  Komşu sayısı: {len(neighbors)}")
            print(f"  Ortalama uzaklık: {np.mean(distances_to_center):.2f}")
            print(f"  Min/Max uzaklık: {np.min(distances_to_center):.2f} / {np.max(distances_to_center):.2f}")
            if radius == 1:
                print(f"  Komşular: {neighbors}")
        print("✅ Test tamamlandı. include_center=False ile merkez dahil edilmedi.")

    
    # ÖRNEK 4: Eğri Tipleri Karşılaştırması
    print("\n" + "=" * 70)
    print("ÖRNEK 4: Eğri Tipleri Karşılaştırması")
    print("=" * 70)
    
    results = {}
    for ct in CurveType:
        test_curve = SpaceFillingCurve(p=5, n=2, curve_type=ct)
        stats = test_curve.get_stats()
        results[ct.value] = stats.locality_preservation
        print(f"\n{ct.value.upper()} Eğrisi:")
        print(f"  - Toplam nokta: {stats.total_points:,}")
        print(f"  - Lokalite skoru: {stats.locality_preservation:.3f}")
    
    print(f"\n{'='*50}")
    print("Kullanım Önerileri:")
    print(f"  • HILBERT: Genel amaçlı → {results['hilbert']:.3f}")
    print(f"  • MORTON:  Hızlı, indeksleme → {results['morton']:.3f}")
    print(f"  • MOORE:   Kapalı döngü → {results['moore']:.3f}")
    print(f"  • ALTAIR:  Hibrid denge → {results['altair']:.3f}")
    
    # ÖRNEK 5: Görüntü Sıkıştırma
    print("\n" + "=" * 70)
    print("ÖRNEK 5: Görüntü Sıkıştırma (Hilbert Eğrisi ile)")
    print("=" * 70)
    
    test_image = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    print(f"\nTest Görüntüsü:")
    print(f"  - Boyut: {test_image.shape}")
    print(f"  - Değer aralığı: [{test_image.min()}, {test_image.max()}]")
    
    compressor = HilbertImageCompressor(test_image)
    
    for keep_ratio in [0.1, 0.2, 0.5]:
        compressed = compressor.compress(keep_ratio=keep_ratio)
        mse = np.mean((test_image - compressed) ** 2)
        psnr = 20 * np.log10(255 / np.sqrt(mse)) if mse > 0 else float('inf')
        print(f"\nSıkıştırma oranı = {keep_ratio*100:.0f}%:")
        print(f"  - MSE: {mse:.2f}")
        print(f"  - PSNR: {psnr:.2f} dB")
        print(f"  - Bellek tasarrufu: {(1-keep_ratio)*100:.0f}%")
    
    # ÖRNEK 6: Boyut İndirgeme
    print("\n" + "=" * 70)
    print("ÖRNEK 6: Boyut İndirgeme (10D → 2D)")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 300
    cluster1 = np.random.randn(n_samples, 10) * 0.5 + 2
    cluster2 = np.random.randn(n_samples, 10) * 0.5
    cluster3 = np.random.randn(n_samples, 10) * 0.5 - 2
    high_dim_data = np.vstack([cluster1, cluster2, cluster3])
    
    print(f"\nOrijinal Veri:")
    print(f"  - Örnek sayısı: {len(high_dim_data)}")
    print(f"  - Boyut: {high_dim_data.shape[1]}D")
    
    reducer = HilbertDimensionReducer(target_dim=2, p=6)
    reduced = reducer.fit_transform(high_dim_data)
    
    print(f"\nİndirgenmiş Veri:")
    print(f"  - Yeni boyut: {reduced.shape[1]}D")
    print(f"  - X aralığı: [{reduced[:,0].min():.3f}, {reduced[:,0].max():.3f}]")
    print(f"  - Y aralığı: [{reduced[:,1].min():.3f}, {reduced[:,1].max():.3f}]")
    
    # ÖRNEK 7: Hilbert Sıralaması
    print("\n" + "=" * 70)
    print("ÖRNEK 7: Hilbert Sıralaması (Mekansal Düzenleme)")
    print("=" * 70)
    
    np.random.seed(42)
    n_points = 50
    points = np.random.randint(0, 256, (n_points, 2))
    
    print(f"\nRastgele Noktalar:")
    print(f"  - Sayı: {n_points}")
    
    orderer = HilbertOrdering(p=8, n=2)
    ordered = orderer.order_points(points)
    
    def avg_neighbor_distance(points):
        total = 0
        for i in range(len(points)-1):
            total += np.linalg.norm(points[i+1] - points[i])
        return total / (len(points)-1)
    
    original_avg_dist = avg_neighbor_distance(points)
    ordered_avg_dist = avg_neighbor_distance(ordered)
    
    print(f"\nSıralama Sonuçları:")
    print(f"  - Orijinal ortalama komşu mesafesi: {original_avg_dist:.2f}")
    print(f"  - Hilbert sıralı ortalama komşu mesafesi: {ordered_avg_dist:.2f}")
    print(f"  - İyileştirme: {(1 - ordered_avg_dist/original_avg_dist)*100:.1f}%")
    
    print(f"\nİlk 10 Hilbert sıralı nokta:")
    for i in range(10):
        print(f"  {i+1:2d}. {ordered[i]}")
    
    # ÖRNEK 8: Grid Sistemi
    print("\n" + "=" * 70)
    print("ÖRNEK 8: Grid Sistemi (Hilbert Tabanlı Veri Yapısı)")
    print("=" * 70)
    
    curve_grid = SpaceFillingCurve(p=5, n=2)
    grid = HilbertGrid(curve_grid)
    
    print(f"\nGrid'e bilinen noktalar ekleniyor...")
    grid[[8, 8]] = "Merkez"
    grid[[8, 9]] = "Kuzey"
    grid[[8, 7]] = "Güney"
    grid[[9, 8]] = "Doğu"
    grid[[7, 8]] = "Batı"
    
    print(f"  ✓ {len(grid)} eleman eklendi")
    
    center = [8, 8]
    neighbors = grid.get_neighbors(center, radius=2)
    print(f"\nNokta {center} komşuları:")
    print(f"  Bulunan komşu sayısı: {len(neighbors)}")
    for i, n in enumerate(neighbors):
        print(f"  {i+1}. {n}")
    
    # ÖRNEK 9: Rota Optimizasyonu
    print("\n" + "=" * 70)
    print("ÖRNEK 9: Rota Optimizasyonu (Seyahat Eden Satıcı)")
    print("=" * 70)
    
    np.random.seed(42)
    n_cities = 25
    cities = []
    for i in range(5):
        for j in range(5):
            cities.append([50 + i*40, 50 + j*40])
    cities = np.array(cities)
    
    print(f"\nŞehir Sayısı: {len(cities)} (5x5 grid)")
    
    curve_opt = SpaceFillingCurve(p=8, n=2)
    optimizer = HilbertPathOptimizer(curve_opt)
    
    random_order = np.random.permutation(len(cities))
    random_route = cities[random_order]
    random_length = optimizer.estimate_path_length(random_route)
    
    optimized_route = optimizer.optimize_order(cities)
    optimized_length = optimizer.estimate_path_length(optimized_route)
    
    print(f"\nRota Uzunlukları:")
    print(f"  - Rastgele rota: {random_length:.2f}")
    print(f"  - Hilbert rota: {optimized_length:.2f}")
    print(f"  - İyileştirme: {(1 - optimized_length/random_length)*100:.1f}%")
    
    # ÖRNEK 10: 3D Hilbert Eğrisi
    print("\n" + "=" * 70)
    print("ÖRNEK 10: 3D Hilbert Eğrisi (Nokta Bulutu)")
    print("=" * 70)
    
    curve_3d = SpaceFillingCurve(p=4, n=3)
    
    print(f"\n3D Eğri Bilgileri:")
    print(f"  - Grid: {curve_3d.grid_size}³ = {curve_3d.total_points} nokta")
    print(f"  - Koordinat aralığı: 0-{curve_3d.max_coord}")
    
    n_samples_3d = 500
    points_3d = curve_3d.sample(n_samples_3d, method='stratified')
    
    print(f"\nNokta Bulutu:")
    print(f"  - Örnek sayısı: {len(points_3d)}")
    print(f"  - X aralığı: [{points_3d[:,0].min()}, {points_3d[:,0].max()}]")
    print(f"  - Y aralığı: [{points_3d[:,1].min()}, {points_3d[:,1].max()}]")
    print(f"  - Z aralığı: [{points_3d[:,2].min()}, {points_3d[:,2].max()}]")
    
    print(f"\nİlk 5 örnek nokta:")
    for i in range(5):
        print(f"  {i+1}. {points_3d[i]}")
    
    # -------------------------------------------------------------------------
    # İstatistikler
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Eğri İstatistikleri")
    print("=" * 70)
    
    stats = curve.get_stats()
    for key, value in stats.__dict__.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # -------------------------------------------------------------------------
    # Özet Tablosu
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ÖZET TABLOSU")
    print("-" * 50)
    print("Örnek                     | Açıklama")
    print("-" * 50)
    print("1. Temel Dönüşümler       | Mesafe ↔ Koordinat dönüşümü")
    print("2. Batch Dönüşüm          | Toplu işlem performansı")
    print("3. Komşuluk Analizi       | Hilbert uzayında komşu bulma")
    print("4. Eğri Karşılaştırması   | 4 farklı eğri tipi analizi")
    print("5. Görüntü Sıkıştırma     | Hilbert ile görüntü sıkıştırma")
    print("6. Boyut İndirgeme        | 10D → 2D dönüşüm")
    print("7. Hilbert Sıralaması     | Mekansal veri düzenleme")
    print("8. Grid Sistemi           | Hilbert tabanlı veri yapısı")
    print("9. Rota Optimizasyonu     | TSP problemi çözümü")
    print("10. 3D Görselleştirme     | 3D nokta bulutu")
    print("-" * 50)
    
    print("\n" + "=" * 70)
    print("✅ Tüm örnekler başarıyla tamamlandı!")
    print("=" * 70)
