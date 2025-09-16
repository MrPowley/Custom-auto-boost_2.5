from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
from typing import Optional

class EncodingFramework(ABC):
    """Encoder framework abstract base class"""
    def __init__(
            self,
            input_path: Path,
            workers: int,
            video_parameters: str = "",
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
        self,
        output_path: Path,
        temp_dir: Path | None = None,
        scenes_path: Path | None = None,
        preset: int |None = None,
        crf: float | None = None,
        ) -> None:
        """Av1an fast pass encoding"""

    @abstractmethod
    def final_pass(
        self,
        output_path: Path,
        preset: int,
        temp_dir: Optional[Path] = None,
        zones_path: Optional[Path] = None,
        ) -> None:
        """Av1an final pass encoding"""

class Av1an(EncodingFramework):
    """Av1an encoding framework"""
    def fast_pass(
            self,
            output_path: Path,
            temp_dir: Path | None = None,
            scenes_path: Path | None = None,
            preset: int | None = None,
            crf: float | None = None,
            ) -> None:
        encoder_parameters = self.fast_pass_encoder_parameters
        if preset is not None:
            encoder_parameters += f" --preset {preset}"
        if crf is not None:
            encoder_parameters += f" --crf {crf:.2f}"

        if self.video_parameters:
            encoder_parameters += f" {self.video_parameters}"

        fast_av1an_command = [
            'av1an',
            '-i', str(self.input_path),
            '-y',
            '--verbose',
            '-m', 'lsmash',
            '-c', 'mkvmerge',
            "-a", "-an", 
            '--min-scene-len', '24',
            '--sc-downscale-height', '720',
            '--set-thread-affinity', '2',
            '-e', self.encoder,
            '-v', encoder_parameters,
            '-w', str(self.workers),
            '-o', str(output_path)
        ]

        if temp_dir:
            fast_av1an_command += ['--temp', str(temp_dir)]
        if scenes_path:
            fast_av1an_command += ['--scenes', str(scenes_path)]

        try:
            subprocess.run(fast_av1an_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Av1an encountered an error:\n{e}")
            exit(1)

    def final_pass(
            self,
            output_path: Path,
            preset: int,
            temp_dir: Optional[Path] = None,
            zones_path: Optional[Path] = None,
            ) -> None:
        video_parameters = self.video_parameters
        video_parameters += f" --preset {preset}"
        # --enable-variance-boost 1 --variance-boost-strength 1 --variance-octile 4 --luminance-qp-bias 10 --qm-min 0 --chroma-qm-min 0

        final_av1an_command = [
            "av1an",
            "-i", str(self.input_path),
            "-y",
            "--split-method", "none",
            "--verbose",
            "-c", "mkvmerge",
            '--force',
            "-e", self.encoder,
            "-v", video_parameters,
            "-o", str(output_path),
            "-w", str(self.workers)
        ]

        if temp_dir is not None:
            final_av1an_command += ["--temp", str(temp_dir)]
        if zones_path is not None:
            final_av1an_command += ["--zones", str(zones_path)]

        try:
            subprocess.run(final_av1an_command, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Av1an encountered an error:\n{e}")
            exit(1)
