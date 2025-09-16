from pathlib import Path
import json
import subprocess
import platform
from abc import ABC, abstractmethod
import os
import re
from statistics import mean, quantiles, median, stdev
import time

import vapoursynth as vs

core = vs.core
core.max_cache_size = 1024

IS_WINDOWS = platform.system() == 'Windows'
NULL_DEVICE = 'NUL' if IS_WINDOWS else '/dev/null'

class Metrics(ABC):
    """Metric abstract class"""
    def __init__(
            self,
            input_path: Path, output_path: Path, skip: int = 1, callback = lambda x: None,
            json_path: Path | None = None, save: bool = False
            ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.json_path = json_path
        self.save = save
        self.skip = skip
        self.callback = callback
        self.scores = []
        self.run_time = 0

    @abstractmethod
    def get_length(self) -> None|int:
        """Returns the input video length in number of frames"""

    @abstractmethod
    def run(self) -> int:
        """Runs the metric calculation"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_run = cls.run

        def timed_run(self, *args, **kwargs):
            start = time.perf_counter()
            result = original_run(self, *args, **kwargs)
            end = time.perf_counter()
            self.run_time = end - start  # store elapsed time in instance
            return result

        cls.run = timed_run

    def save_scores(self, metric: str):
        """Saves the Metric scores in a json format"""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump({"skip": self.skip, metric: self.scores}, file)

    @staticmethod
    def get_metric(json_path: Path, metric: str) -> tuple[list[float], int]:
        """Reads from a json file and returns the metric scores and skip values"""
        scores: list[float] = []

        with open(json_path, "r", encoding="utf-8") as file:
            content = json.load(file)
            skip: int = content["skip"]
            scores = content[metric]

        return scores, skip

    def calculate_metrics_stats(self) -> tuple[float, float, float, float, float, float, float]:
        """
        Takes a list of metrics scores and returns the associated arithmetic mean,
        5th percentile and 95th percentile scores.

        :param score_list: list of SSIMU2 scores
        :type score_list: list
        """

        scores_minimum = min(self.scores)
        self.scores = [score if score >= 0 else 0.0 for score in self.scores]

        scores_average = mean(self.scores)
        scores_maximum = max(self.scores)
        scores_percentiles = quantiles(self.scores, n=100)
        scores_low_percentile = scores_percentiles[4]
        scores_high_percentile = scores_percentiles[94]
        scores_median = median(self.scores)
        scores_standard_deviation = stdev(self.scores)

        return (
            scores_average, scores_standard_deviation,
            scores_median, scores_low_percentile,
            scores_high_percentile, scores_minimum, scores_maximum
        )

class VSMetrics(Metrics):
    """VS Metrics class"""
    def __init__(
            self,
            input_path: Path,
            output_path: Path,
            json_path: Path | None = None,
            skip: int = 1,
            callback = lambda x: None,
            save = False,
            ) -> None:
        super().__init__(input_path, output_path, skip, callback, json_path, save)
        self.source_clip = None
        self.encoded_clip = None

        self.cut_source_clip = None
        self.cut_encoded_clip = None

        self.load_clips()
        self.cut_clips()

    def load_clips(self) -> None:
        """Load clips as vs objects and handles vpy input scripts"""
        is_vpy = self.input_path.suffix == ".vpy"
        vpy_vars = {}
        if is_vpy:
            exec(open(self.input_path).read(), globals(), vpy_vars)

        if not is_vpy:
            self.source_clip: vs.VideoNode = core.lsmas.LWLibavSource(source=self.input_path, cache=0)
        else:
            self.source_clip: vs.VideoNode = vpy_vars["clip"]
        self.encoded_clip: vs.VideoNode = core.lsmas.LWLibavSource(source=self.output_path, cache=0)

    def cut_clips(self):
        """Cuts the clips every <skip> frames"""
        if self.skip > 1:
            self.cut_source_clip = self.source_clip[::self.skip]
            self.cut_encoded_clip = self.encoded_clip[::self.skip]
        else:
            self.cut_source_clip = self.source_clip
            self.cut_encoded_clip = self.encoded_clip

    def get_length(self) -> int:
        return len(self.source_clip)

    def get_duration(self) -> float:
        """Calculates and returns the input video duration"""
        length = self.get_length()
        fps = self.source_clip.fps.numerator / self.source_clip.fps.denominator
        return round(length / fps, 3)


class VSzipXPSNR(VSMetrics):
    """VSZIP XPSNR class"""
    def run(self):
        """Runs the vszip XPSNR calculation and weights the scores"""
        result = core.vszip.XPSNR(self.cut_source_clip, self.cut_encoded_clip)
        for frame in result.frames():
            y = frame.props['XPSNR_Y'] if frame.props['XPSNR_Y'] != float("inf") else 100.0
            u = frame.props['XPSNR_U'] if frame.props['XPSNR_U'] != float("inf") else 100.0
            v = frame.props['XPSNR_V'] if frame.props['XPSNR_V'] != float("inf") else 100.0
            w = (4 * y + u + v) / 6

            self.scores.append(w)
            self.callback(self.skip)

        if self.save:
            self.save_scores("XPSNR")
        return 0

class VSzipSSIMULACRA2(VSMetrics):
    """VSZIP SSIMULACRA2 class"""
    def run(self):
        """Runs the vszip SSIMULACRA2 calculation"""
        result = core.vszip.SSIMULACRA2(self.cut_source_clip, self.cut_encoded_clip)

        for frame in result.frames():
            score = frame.props['SSIMULACRA2']
            self.scores.append(score)
            self.callback(self.skip)

        if self.save:
            self.save_scores("SSIMULACRA2")
        return 0

class VShipSSIMULACRA2(VSMetrics):
    """VSHIP SSIMULACRA class"""
    def run(self):
        """Runs the vship SSIMULACRA calculation"""
        try:
            result = core.vship.SSIMULACRA2(
                self.cut_source_clip, self.cut_encoded_clip, numStream=4)

            for frame in result.frames():
                score = frame.props['_SSIMULACRA2']
                self.scores.append(score)
                self.callback(self.skip)

            if self.save:
                self.save_scores("SSIMULACRA2")
        except AttributeError:
            print("\nvship plugin not installed\n")
            return 1
        return 0

class TurboMetricsSSIMULACRA2(Metrics):
    """Turbo Metrics SSIMULACRA2"""
    def run(self):
        """Runs the turbo-metrics calculation"""
        turbo_cmd = [
            "turbo-metrics",
            "-m", "ssimulacra2", # Metric
            "--output", "json-lines",
            "--every", str(self.skip),
            str(self.input_path), str(self.output_path)
        ]

        try:
            turbo_process = subprocess.Popen(
                turbo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print("\nTurbo-metrics not found\n")
            return 1

        for line in turbo_process.stdout: # pyright: ignore[reportOptionalIterable]
            content = json.loads(line)
            if not "frame_count" in content:
                score = content["ssimulacra2"] # Metric
                self.scores.append(score)
                self.callback(self.skip)

        for line in turbo_process.stderr: # pyright: ignore[reportOptionalIterable]
            if "MkvUnknownCodec" in line:
                print("\nTurbo-metrics has failed." \
                      "t only supports av1, h264 and h262 input codecs\n")
                return 1

        turbo_process.wait()
        # returncode = turbo_process.returncode

        if self.save:
            self.save_scores("SSIMULACRA2")
        return 0

    def get_length(self) -> None | int:
        return None

class FFmpegXPSNR(Metrics):
    """FFmpeg XPSNR class"""
    def __init__(
            self,
            input_path: Path, output_path: Path, skip: int = 1, callback = lambda x: None,
            json_path: Path | None = None, save: bool = False
            ) -> None:
        super().__init__(input_path, output_path, skip, callback, json_path, save)
        if IS_WINDOWS:
            self.xpsnr_tmp_stats_path = Path("xpsnr.log")
            src_file_dir = input_path.parent
            os.chdir(src_file_dir)
        else:
            self.xpsnr_tmp_stats_path = Path("xpsnr.log")

    def run(self) -> int:
        xpsnr_command = [
            'ffmpeg',
            '-i', str(self.input_path),
            '-i', str(self.output_path),
            '-lavfi', f'xpsnr=stats_file={str(self.xpsnr_tmp_stats_path)}',
            '-f', 'null', NULL_DEVICE
        ]

        previous_frame_progress = 0

        try:
            xpsnr_process = subprocess.Popen(xpsnr_command, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT,universal_newlines=True)
            for line in xpsnr_process.stdout: # pyright: ignore[reportOptionalIterable]
                match = re.search(r'frame=\s*(\d+)', line)
                if match:
                    current_frame_progress = int(match.group(1))
                    delta = current_frame_progress - previous_frame_progress
                    previous_frame_progress = current_frame_progress
                    self.callback(delta)

        except subprocess.CalledProcessError as e:
            print(f'XPSNR encountered an error:\n{e}')
            return 1

        self.scores = self.evaluate_xpsnr_log(self.xpsnr_tmp_stats_path)

        if self.save:
            self.save_scores("XPSNR")

        self.xpsnr_tmp_stats_path.unlink()

        return 0

    def evaluate_xpsnr_log(self, xpsnr_log_path: Path) -> list[float]:
        """Reads XPSNR log file and returns the list of weighted values"""
        values_weighted: list[float] = []
        with open(xpsnr_log_path, "r") as file:
            for line in file.readlines():
                match = re.search(
                    r"XPSNR [yY]: ([0-9]+\.[0-9]+|inf)  XPSNR [uU]: ([0-9]+\.[0-9]+|inf)  XPSNR [vV]: ([0-9]+\.[0-9]+|inf)",
                    line)
                if match:
                    y = float(match.group(1)) if match.group(1) != 'inf' else 100.0
                    u = float(match.group(2)) if match.group(2) != 'inf' else 100.0
                    v = float(match.group(3)) if match.group(3) != 'inf' else 100.0
                    w = (4 * y + u + v) / 6
                    values_weighted.append(w)

        average_weighted = sum(values_weighted) / len(values_weighted)

        values_weighted_averaged = [value_weighted / average_weighted
                                    for value_weighted in values_weighted]

        return values_weighted_averaged

    def get_length(self) -> None | int:
        return None
