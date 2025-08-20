from tqdm import tqdm

class ProgressBar:
    """General metric progressbar"""
    def __init__(self) -> None:
        self.pbar = None

    def initialize_progressbar(self, total=None, description=None, unit=" frames", smoothing=0):
        """Instanciates a tqdm object"""
        self.pbar = tqdm(total=total, desc=description, unit=unit, smoothing=smoothing)

    def update_progressbar(self, increment):
        """Updates the progressbar by specific increment"""
        self.pbar.update(increment)
