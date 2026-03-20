"""Benchmark color_blend performance: nested loop vs vectorized."""
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.color_blend import color_blend

def benchmark_color_blend():
    """Benchmark color blend on realistic image sizes."""
    sizes = [256, 512, 1024]
    results = {}
    
    for size in sizes:
        # Create test images
        base = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        blend = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
        
        # Warm up (compile any JIT)
        color_blend(base, blend)
        
        # Benchmark
        iters = 5 if size <= 512 else 2
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            color_blend(base, blend)
            times.append(time.perf_counter() - t0)
        
        avg_ms = np.mean(times) * 1000
        results[size] = avg_ms
        pixels = size * size
        throughput = (pixels / (avg_ms / 1000)) / 1e6  # megapixels/sec
        print(f"{size}x{size} ({pixels:,} pixels): {avg_ms:.2f}ms ({throughput:.1f} MP/s)")
    
    return results

if __name__ == '__main__':
    print("=== Color Blend Performance (Vectorized) ===\n")
    benchmark_color_blend()
