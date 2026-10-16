def format_duration(
    duration: float,
) -> str:
    minute, second = divmod(round(duration), 60)
    return f"{minute}m{second:02d}s"


def format_memory(
    num_bytes: int,
) -> str:
    return f"{num_bytes / 2**30:.1f}GB"
