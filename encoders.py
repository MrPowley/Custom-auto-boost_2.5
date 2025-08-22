from abc import ABC, abstractmethod
from pathlib import Path
import subprocess

class EncodingFramework(ABC):
    """Encoder framework abstract base class"""
    def __init__(
            self,
            input_path: Path,
            workers: int,
            video_parameters: str,
            encoder: str = "svt-av1"
            ) -> None:
        self.input_path = input_path
        self.encoder = encoder
        self.workers = workers
        self.video_parameters = video_parameters

        self.fast_pass_encoder_parameters = '--lp 2 --keyint 0 --scm 0' \
                                   ' --fast-decode 1 --color-primaries 1' \
                                   ' --transfer-characteristics 1 --matrix-coefficients 1'


    @abstractmethod
    def fast_pass(
        self, output_path: Path, temp_dir: Path, scenes_path: Path, preset: int, crf: float
        ):
        """Av1an fast pass encoding"""

    @abstractmethod
    def final_pass(self, output_path: Path, temp_dir: Path, zones_path: Path, preset: int):
        """Av1an final pass encoding"""

class Av1an(EncodingFramework):
    """Av1an encoding framework"""
    def fast_pass(
            self, output_path: Path, temp_dir: Path, scenes_path: Path, preset: int, crf: float):
        encoder_parameters = self.fast_pass_encoder_parameters
        encoder_parameters += f" --preset {preset} --crf {crf:.2f}"

        if self.video_parameters:
            encoder_parameters += f" {self.video_parameters}"

        fast_av1an_command = [
            'av1an',
            '-i', str(self.input_path),
            '--temp', str(temp_dir),
            '-y',
            '--verbose',
            '-m', 'lsmash',
            '-c', 'mkvmerge',
            '--min-scene-len', '24',
            '--scenes', str(scenes_path),
            '--sc-downscale-height', '720',
            '--set-thread-affinity', '2',
            '-e', 'svt-av1',
            '-v', encoder_parameters,
            '-w', str(self.workers),
            '-o', str(output_path)
        ]

        try:
            subprocess.run(fast_av1an_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Av1an encountered an error:\n{e}")
            exit(1)

    def final_pass(self, output_path: Path, temp_dir: Path, zones_path: Path, preset: int):
        video_parameters = self.video_parameters
        video_parameters += f"--preset {preset}"

        final_av1an_command = [
            "av1an",
            "-i", str(self.input_path),
            "--temp", str(temp_dir),
            "-y",
            "--split-method", "none",
            "--verbose",
            "-c", "mkvmerge",
            '--force',
            "-e", "svt-av1",
            "-v", video_parameters,
            "--zones", str(zones_path),
            "-o", str(output_path),
            "-w", str(self.workers)
        ]

        try:
            subprocess.run(final_av1an_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Av1an encountered an error:\n{e}")
            exit(1)




# av1an = Av1an(Path("G:/noamh/Code/Custom-auto-boost-2.5/test/sample.mkv"), "svt-av1", 8, "")
# av1an.fast_pass(Path("G:/noamh/Code/Custom-auto-boost-2.5/test/out.mkv"), Path("G:/noamh/Code/Custom-auto-boost-2.5/test/sample"), Path("G:/noamh/Code/Custom-auto-boost-2.5/test/sample/scenes.json"), 8, 30.0)
