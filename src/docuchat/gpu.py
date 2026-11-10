from docuchat.config import settings

try:
    import spaces
    ZERO_GPU = True
except ImportError:
    ZERO_GPU = False

USE_GPU = settings.use_gpu
DEVICE = "cuda" if USE_GPU else "cpu"

if ZERO_GPU:
    gpu = spaces.GPU(duration=60) if USE_GPU else (lambda fn: fn)

    if not USE_GPU:
        @spaces.GPU
        def _gpu_placeholder():
            return True
else:
    gpu = lambda fn: fn