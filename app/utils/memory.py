"""Process memory telemetry (RSS)."""


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return round(memory_info.rss / (1024 * 1024), 2)
    except ImportError:
        return 0.0
    except Exception:
        return 0.0
