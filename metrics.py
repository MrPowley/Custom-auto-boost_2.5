from pathlib import Path
import json
import subprocess
# import shutil

import vapoursynth as vs

core = vs.core
core.max_cache_size = 1024

class VSMetrics:
    """Vapoursynth metrics calculation"""
    def __init__(
            self, source_file: Path, encoded_file: Path, json_path: Path = None,
            skip: int = 1, update_callback_func = lambda x: None) -> None:
        self.source_file = source_file
        self.encoded_file = encoded_file
        self.json_path = json_path
        self.skip = skip
        self.callback = update_callback_func

        self.source_clip = None
        self.encoded_clip = None
        self.cut_source_clip = None
        self.cut_encoded_clip = None

        self.scores = []

        self.post_init()

    def post_init(self):
        """After init method"""
        self.load_clips()
        self.cut_clips()

    def set_json_path(self, json_path: Path):
        """Set json path"""
        self.json_path = json_path

    def load_clips(self) -> None:
        """Load clips as vs objects and handles vpy input scripts"""
        is_vpy = self.source_file.suffix == ".vpy"
        vpy_vars = {}
        if is_vpy:
            exec(open(self.source_file).read(), globals(), vpy_vars)

        if not is_vpy:
            self.source_clip = core.lsmas.LWLibavSource(source=self.source_file, cache=0)
        else:
            self.source_clip = vpy_vars["clip"]
        self.encoded_clip = core.lsmas.LWLibavSource(source=self.encoded_file, cache=0)

    def cut_clips(self):
        """Cuts the clips every <skip> frames"""
        if self.skip > 1:
            self.cut_source_clip = self.source_clip.std.SelectEvery(cycle=self.skip, offsets=1)
            self.cut_encoded_clip = self.encoded_clip.std.SelectEvery(cycle=self.skip, offsets=1)
        else:
            self.cut_source_clip = self.source_clip
            self.cut_encoded_clip = self.encoded_clip

    def get_length(self) -> int:
        """Returns the number of frames of the source clip"""
        return len(self.source_clip)

    def xpsnr_vszip(self):
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

    def ssimulacra2_vszip(self):
        """Runs the vszip SSIMULACRA2 calculation"""
        result = core.vszip.SSIMULACRA2(self.cut_source_clip, self.cut_encoded_clip)

        for frame in result.frames():
            score = frame.props['SSIMULACRA2']
            self.scores.append(score)
            self.callback(self.skip)

        self.save_scores("SSIMULACRA2")

    def ssimulcara2_vship(self):
        """Runs the vship SSIMULACRA calculation"""
        result = core.vship.SSIMULACRA2(self.cut_source_clip, self.cut_encoded_clip)

        for frame in result.frames():
            score = frame.props['_SSIMULACRA2']
            self.scores.append(score)
            self.callback(self.skip)

        self.save_scores("SSIMULACRA2")

    def save_scores(self, metric):
        """Saves the XPSNR scores in a json format"""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump({"skip": self.skip, metric: self.scores}, file)

class TurboMetrics:
    """Turbo-metrics SSIMULACRA2 calculation"""
    def __init__(
            self, source_file: Path, encoded_file: Path, json_path: Path = None,
            skip: int = 1, update_callback_func = lambda x: None) -> None:
        self.source_file = source_file
        self.encoded_file = encoded_file
        self.json_path = json_path
        self.skip = skip
        self.callback = update_callback_func

        self.scores = []

    def run(self, metric: str):
        """Runs the turbo-metrics calculation"""
        turbo_cmd = [
            "turbo-metrics",
            "-m", metric,
            "--output", "json-lines",
            "--every", str(self.skip),
            str(self.source_file), str(self.encoded_file)
        ]

        turbo_process = subprocess.Popen(
            turbo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        for line in turbo_process.stdout:
            content = json.loads(line)
            if not "frame_count" in content:
                score = content[metric]
                self.scores.append(score)
                self.callback(self.skip)

        for line in turbo_process.stderr:
            if "MkvUnknownCodec" in line:
                raise Exception(
                    "Turbo-metrics has failed. It only supports av1, h264 and h262 input codecs")

        turbo_process.wait()
        # returncode = turbo_process.returncode

        self.save_scores(metric)

    def get_length(self):
        """(WIP) Supposed to return the length of the video"""
        return None

    def save_scores(self, metric: str):
        """Saves the turbo-metrics scores in a json format"""
        with open(self.json_path, "w", encoding="utf-8") as file:
            json.dump({"skip": self.skip, metric: self.scores}, file)
