from rich import progress

def progress_bar() -> progress.Progress:
    return progress.Progress(
        '[progress.description]{task.description}',
        progress.BarColumn(),
        '[progress.percentage]{task.percentage:>3.0f}%',
        progress.TimeRemainingColumn(),
        progress.TimeElapsedColumn(),
        transient=True,
    )
