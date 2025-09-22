import time, threading
try:
    import pynvml
    pynvml.nvmlInit()
    _h = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _h = None

def read_power_w():
    if _h is None: return None
    return pynvml.nvmlDeviceGetPowerUsage(_h) / 1000.0  # mW -> W

def read_mem_mb():
    if _h is None: return None, None
    info = pynvml.nvmlDeviceGetMemoryInfo(_h)
    return info.used / (1024**2), info.total / (1024**2)

def read_temp_c():
    if _h is None: return None
    return pynvml.nvmlDeviceGetTemperature(_h, pynvml.NVML_TEMPERATURE_GPU)

def read_clock_mhz():
    if _h is None: return None
    try:
        return pynvml.nvmlDeviceGetClockInfo(_h, pynvml.NVML_CLOCK_SM)
    except Exception:
        return None

class PowerLogger:
    def __init__(self, interval_s=0.05, warmup_s=60.0, vram_interval_s=1.0):
        self.interval = interval_s
        self.warmup_s = warmup_s
        self.vram_interval_s = vram_interval_s
        self.samples = []   # (t, W)
        self.vram = []      # (t, used_MB)
        self._stop = threading.Event()
        self._thr = None
        self._thr_vram = None

    def _loop_power(self):
        while not self._stop.is_set():
            p = read_power_w()
            if p is not None:
                self.samples.append((time.time(), p))
            time.sleep(self.interval)

    def _loop_vram(self):
        while not self._stop.is_set():
            u,_ = read_mem_mb()
            if u is not None:
                self.vram.append((time.time(), u))
            time.sleep(self.vram_interval_s)

    def start(self):
        self._thr = threading.Thread(target=self._loop_power, daemon=True)
        self._thr.start()
        self._thr_vram = threading.Thread(target=self._loop_vram, daemon=True)
        self._thr_vram.start()

    def stop(self):
        self._stop.set()
        if self._thr: self._thr.join()
        if self._thr_vram: self._thr_vram.join()

    def integrate_energy_j(self):
        pts = self.samples
        if len(pts) < 2: return 0.0, 0.0, 0.0
        t0 = pts[0][0] + self.warmup_s
        pts = [(t,p) for (t,p) in pts if t >= t0]
        if len(pts) < 2: return 0.0, 0.0, 0.0
        E = 0.0
        for (t1,p1),(t2,p2) in zip(pts[:-1], pts[1:]):
            E += 0.5*(p1+p2)*(t2-t1)
        dur = pts[-1][0] - pts[0][0]
        avg_p = E/dur if dur>0 else 0.0
        peak_p = max(p for _,p in pts)
        return E, avg_p, peak_p

    def vram_stats(self):
        if not self.vram: return None, None
        vals = [u for _,u in self.vram]
        return max(vals), sum(vals)/len(vals)
