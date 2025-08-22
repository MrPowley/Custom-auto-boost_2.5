from pathlib import Path
import json
import subprocess
# import shutil
from abc import ABC, abstractmethod

import vapoursynth as vs

core = vs.core
core.max_cache_size = 1024

class Metrics(ABC):
    """Metric abstract class"""
    def __init__(
            self, input_path: Path, output_path: Path, json_path: Path, skip: int, callback
            ) -> None:
        self.input_path = input_path
        self.output_path = output_path
        self.json_path = json_path
        self.skip = skip
        self.callback = callback
        self.scores = []

    @abstractmethod
    def get_length(self) -> None|int:
        """Returns the input video length in number of frames"""

    @abstractmethod
    def run(self) -> int:
        """Runs the metric calculation"""

    def save_scores(self, metric: str):
        """Saves the Metric scores in a json format"""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump({"skip": self.skip, metric: self.scores}, file)

class VSMetrics(Metrics):
    """VS Metrics class"""
    def __init__(
            self, input_path: Path, output_path: Path, json_path: Path, skip: int, callback
            ) -> None:
        super().__init__(input_path, output_path, json_path, skip, callback)
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
            self.source_clip = core.lsmas.LWLibavSource(source=self.input_path, cache=0)
        else:
            self.source_clip = vpy_vars["clip"]
        self.encoded_clip = core.lsmas.LWLibavSource(source=self.output_path, cache=0)

    def cut_clips(self):
        """Cuts the clips every <skip> frames"""
        if self.skip > 1:
            self.cut_source_clip = self.source_clip.std.SelectEvery(cycle=self.skip, offsets=1)
            self.cut_encoded_clip = self.encoded_clip.std.SelectEvery(cycle=self.skip, offsets=1)
        else:
            self.cut_source_clip = self.source_clip
            self.cut_encoded_clip = self.encoded_clip

    def get_length(self):
        return len(self.source_clip)

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

        self.save_scores("SSIMULACRA2")
        return 0

class VShipSSIMULACRA2(VSMetrics):
    """VSHIP SSIMULACRA class"""
    def run(self):
        """Runs the vship SSIMULACRA calculation"""
        try:
            result = core.vship.SSIMULACRA2(self.cut_source_clip, self.cut_encoded_clip)

            for frame in result.frames():
                score = frame.props['_SSIMULACRA2']
                self.scores.append(score)
                self.callback(self.skip)

            self.save_scores("SSIMULACRA2")
        except AttributeError:
            print("\nvship plugin not installed\n")
            return 1
        return 0

class TurboMetricsSSIMULACRA2(Metrics):
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

        for line in turbo_process.stdout:
            content = json.loads(line)
            if not "frame_count" in content:
                score = content["ssimulacra2"] # Metric
                self.scores.append(score)
                self.callback(self.skip)

        for line in turbo_process.stderr:
            if "MkvUnknownCodec" in line:
                print("\nTurbo-metrics has failed. It only supports av1, h264 and h262 input codecs\n")
                return 1

        turbo_process.wait()
        # returncode = turbo_process.returncode

        self.save_scores("SSIMULACRA2")
        return 0

    def get_length(self) -> None | int:
        return None
