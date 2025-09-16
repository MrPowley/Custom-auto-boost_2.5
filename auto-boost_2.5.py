# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tqdm",
#   "psutil",
#   "vapoursynth",
# ]
# ///

# Originally by Trix
# Contributors: R1chterScale, Yiss, Kosaka & others from AV1 Weeb edition
# Modified by MrPowley
# License: MIT

from math import ceil
from pathlib import Path
import json
import argparse
from typing import Optional
from statistics import mean, quantiles

import psutil
from rich.console import Console

import ui
import metrics
import encoders

console = Console()
verbose = False

class AutoBoost:
    """Auto boost class"""
    def __init__(
            self,
            input_path: Path,
            output_path: Optional[Path] = None,
            temp_dir: Optional[Path] = None,
            workers: Optional[int] = None,
            video_parameters: str = '',
            preset: Optional[int] = None,
            fastpass_preset: Optional[int] = 6,
            finalpass_preset: Optional[int] = 4,
            crf: float = 30.0,
            method: int = 3,
            metrics_implementations: str = 'vship,vszip',
            skip: Optional[int] = None,
            ssimulacra2_skip: Optional[int] = None,
            xpsnr_skip: Optional[int] = None,
            base_deviation: float = 10.0,
            max_positive_deviation: Optional[float] = None,
            max_negative_deviation: Optional[float] = None,
            aggressiveness: float = 20.0,
            ) -> None:
        self.input_path = input_path
        self.video_parameters = video_parameters
        self.crf = crf
        self.metrics_implementations = metrics_implementations
        self.base_deviation = base_deviation
        self.max_positive_deviation = max_positive_deviation
        self.max_negative_deviation = max_negative_deviation
        self.aggressiveness = aggressiveness

        self.ranges = [0]
        self.average = 0.0
        self.percentile_5_total = []

        self.av1an: encoders.EncodingFramework | None = None

        self.method = method
        match self.method:
            case 1:
                self.metric = 'ssimulacra2'
            case 2:
                self.metric = 'xpsnr'
            case 3:
                self.metric = 'multiplied'
            case 4:
                self.metric = 'minimum'

        if fastpass_preset is not None:
            self.fastpass_preset = fastpass_preset
        else:
            if preset is not None:
                self.fastpass_preset = preset
            else:
                self.fastpass_preset = 6

        if finalpass_preset is not None:
            self.finalpass_preset = finalpass_preset
        else:
            if preset is not None:
                self.finalpass_preset = preset
            else:
                self.finalpass_preset = 4

        # Workers
        if workers is not None:
            self.workers: int = workers
        else:
            count: int | None = psutil.cpu_count(logical=False)
            if count is None:
                self.workers: int = 1
            else:
                self.workers: int = count

        if ssimulacra2_skip is not None:
            self.ssimulacra2_skip = ssimulacra2_skip
        if xpsnr_skip is not None:
            self.xpsnr_skip = xpsnr_skip

        if ssimulacra2_skip is None:
            self.ssimulacra2_skip = 3
        if xpsnr_skip is None:
            self.xpsnr_skip = 1

        if skip is not None and self.ssimulacra2_skip is None:
            self.ssimulacra2_skip: int = skip
        if skip is not None and self.xpsnr_skip is None:
            self.xpsnr_skip: int = skip

        self.output_dir: Path = self.input_path.parent

        # Output path
        if output_path is not None:
            self.output_path: Path = output_path
        else:
            self.output_path : Path = self.output_dir / f"{input_path.stem}_boosted.mkv"

        # Temp dir
        if temp_dir is not None:
            self.temp_dir: Path = self.temp_dir.resolve()
        else:
            self.temp_dir: Path = self.input_path.parent / self.input_path.stem

        self.fastpass_output_path: Path = self.temp_dir / "fastpass.mkv"
        self.fastpass_temp_dir: Path = self.temp_dir / "fastpass"
        self.finalpass_temp_dir: Path = self.temp_dir / "finalpass"
        self.zones_path: Optional[Path] = self.temp_dir / f'{self.metric}_zones.txt'
        self.scenes_path: Optional[Path] = self.temp_dir / "scenes.json"

        self.metric_implementation = (
            self.resolve_implementation(
                self.metrics_implementations,
                self.method,
                self.ssimulacra2_skip,
                self.xpsnr_skip
            )
        )

        self.ssimulacra2_implementation = self.metric_implementation['ssimulacra2']['implementation']
        self.ssimulacra2_skip = self.metric_implementation['ssimulacra2']['skip']

        self.xpsnr_implementation = self.metric_implementation['xpsnr']['implementation']
        self.xpsnr_skip = self.metric_implementation['xpsnr']['skip']

        self.ssimu2_json_path = self.temp_dir / f"{self.input_path.stem}_ssimulacra2.json"
        self.xpsnr_json_path = self.temp_dir / f"{self.input_path.stem}_xpsnr.json"

    def stage1(self) -> None:
        """fastpass() method alias"""
        self.fastpass()

    def stage2(self) -> None:
        """measure_metrics() method alias"""
        self.measure_metrics()

    def stage3(self) -> None:
        """boost() method alias"""
        self.boost()

    def stage4(self) -> None:
        """finalpass() method alias"""
        self.finalpass()

    def run_all(self) -> None:
        """Runs all stages of AutoBoost"""
        self.fastpass()
        self.measure_metrics()
        self.boost()
        self.finalpass()

    def fastpass(self) -> None:
        """Runs stage 1"""
        av1an = encoders.Av1an(self.input_path, self.workers, self.video_parameters)
        av1an.fast_pass(
            self.fastpass_output_path,
            self.fastpass_temp_dir,
            self.scenes_path,
            self.fastpass_preset,
            self.crf
            )

    def measure_metrics(self) -> None:
        """Runs stage 2"""

        match self.method:
            case 1:
                self.calculate_ssimulacra2()
            case 2:
                self.calculate_xpsnr()
            case 3|4:
                self.calculate_ssimulacra2()
                self.calculate_xpsnr()

    def boost(self) -> None:
        """Runs stage 3"""
        self.get_ranges()
        self.calculate_zones()

    def finalpass(self) -> None:
        """Runs stage 4"""
        if self.zones_path is not None and not self.zones_path.exists():
            self.zones_path = None

        av1an = encoders.Av1an(self.input_path, self.workers, self.video_parameters)
        av1an.final_pass(
            self.output_path,
            self.finalpass_preset,
            self.finalpass_temp_dir,
            self.zones_path,
            )

    def resolve_implementation(
            self,
            string: str,
            method: int,
            ssimulacra2_skip: int,
            xpsnr_skip: int
        ) -> dict:
        """Handles the metric-implementation arg"""
        implementations = string.split(",")
        if len(implementations) < 2:
            implementations += [""]

        if method == 2:
            ssimulacra2_index = 1
            xpsnr_index = 0
        else:
            ssimulacra2_index = 0
            xpsnr_index = 1

        ssimulacra2_skip_value = ssimulacra2_skip if ssimulacra2_skip is not None else 1
        xpsnr_skip_value = xpsnr_skip if xpsnr_skip is not None else 1

        implementations_dict = {
            "ssimulacra2": {
                "implementation": implementations[ssimulacra2_index],
                "skip": ssimulacra2_skip_value
                },
            "xpsnr": {
                "implementation": implementations[xpsnr_index],
                "skip": xpsnr_skip_value
                }
        }

        # Handle SSIMULACRA2
        if (
            not implementations_dict["ssimulacra2"]["implementation"]
            or implementations_dict["ssimulacra2"]["implementation"]
            not in ("vszip", "vship", "turbo-metrics")
            ):
            implementations_dict["ssimulacra2"]["implementation"] = "vship"

        # Handle xpsnr
        if (
            not implementations_dict["xpsnr"]["implementation"]
            or implementations_dict["xpsnr"]["implementation"] not in ("vszip", "ffmpeg")
            ):
            implementations_dict["xpsnr"]["implementation"] = "vszip"

        # Add skip for ssimulacra2 vszip
        if (implementations_dict["ssimulacra2"]["implementation"] == "vszip"
            and not ssimulacra2_skip):
            implementations_dict["ssimulacra2"]["skip"] = 3

        return implementations_dict

    def calculate_ssimulacra2(self) -> None:
        """Handles implementation and calls the appropriated SSIMULACRA2 calculation method"""

        implementation_handlers: dict[str, list[metrics.Metrics]] = {
            "turbo-metrics": [
                metrics.TurboMetricsSSIMULACRA2,
                metrics.VShipSSIMULACRA2,
                metrics.VSzipSSIMULACRA2],
            "vship": [
                metrics.VShipSSIMULACRA2,
                metrics.TurboMetricsSSIMULACRA2,
                metrics.VSzipSSIMULACRA2],
            "vszip": [
                metrics.VSzipSSIMULACRA2]
        } # pyright: ignore[reportAssignmentType]

        for handler in implementation_handlers.get(
            self.metric_implementation['ssimulacra2']['implementation']):
            print(f"Running {handler}")
            metric = self.init_metric(
                self.ssimu2_json_path,
                self.metric_implementation['ssimulacra2']['skip'],
                handler,
                'SSIMULACRA2'
                )
            returncode = metric.run()
            if returncode == 0:
                return None
        raise RuntimeError("All SSIMULACRA2 metric implementations failed")

    def calculate_xpsnr(self) -> None:
        """Handles implementation and calls the appropriated XPSNR calculation method"""

        implementation_handlers = {
            'vszip': [metrics.VSzipXPSNR],
            'ffmpeg': [metrics.FFmpegXPSNR, metrics.VSzipXPSNR]
        }

        for handler in implementation_handlers.get(
            self.metric_implementation['xpsnr']['implementation']):
            print(f'Running {handler}')
            metric = self.init_metric(
                self.xpsnr_json_path,
                self.metric_implementation['xpsnr']['skip'],
                handler,
                'XPSNR'
                )
            returncode = metric.run()
            if returncode == 0:
                return None
        raise RuntimeError("All XPSNR metric implementations failed")

    def init_metric(
            self,
            json_path: Path,
            skip: int,
            metric_class,
            metric_name: str
            ):
        """Initializes the metric and its progressbar"""
        progressbar = ui.ProgressBar()
        metric = metric_class(
            self.input_path,
            self.fastpass_output_path,
            json_path,
            skip,
            progressbar.update_progressbar,
            True
            )

        video_length = metric.get_length()

        progressbar.initialize_progressbar(
            total=video_length, description=f"Calculating {metric_name}")

        return metric

    def get_ranges(self) -> None:
        """
        Reads a scene file and returns a list of frame numbers for each scene change.

        Args:
            scenes (Path): scenes.json path

        Returns:
            List of ranges (int)
        """

        with open(self.scenes_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
            for scene in content['scenes']:
                self.ranges.append(scene['end_frame'])

    def calculate_zones(self) -> None:
        """Calcules zones with chosen method and metric"""
        match self.method:
            case 1 | 2:
                if self.method == 1:
                    metric = 'SSIMULACRA2'
                else:
                    metric = 'XPSNR'

                metric_json_path = self.temp_dir / f'{self.input_path.stem}_{metric}.json'
                metric_scores, skip = self.get_metric(metric_json_path, metric)

                # Expand the scores list with dummy values
                #  to the full length of the video to compensate the skip
                metric_scores_expanded = []
                for score in metric_scores:
                    metric_scores_expanded += [score] + [-1] * (skip - 1)

                total_scores = []
                self.percentile_5_total = []
                for i in range(len(self.ranges) - 1):
                    if metric_scores_expanded[self.ranges[i]:self.ranges[i+1]]:
                        chunk_scores_expanded = metric_scores_expanded[
                            self.ranges[i]:self.ranges[i+1]
                            ]
                        # Remove the dummy values
                        chunk_scores = [score for score in chunk_scores_expanded if score != -1]
                        # print(chunk_scores, chunk_scores_expanded)
                        chunk_percentile_5 = quantiles(chunk_scores, n=100)[4]
                        self.percentile_5_total.append(chunk_percentile_5)
                        total_scores += chunk_scores

                self.average = mean(total_scores)

                # print(f'{metric}')
                # print(f'Median score: {metric_average}')
                # print(f'5th Percentile: {metric_percentile_5}')
                # print(f'95th Percentile: {metric_percentile_95}')
                self.generate_zones()

            case 3 | 4:
                if self.method == 3:
                    method_name = 'multiplied'
                else:
                    method_name = 'minimum'

                ssimu2_json_path = self.temp_dir / f"{self.input_path.stem}_ssimulacra2.json"
                ssimu2_scores, skip = self.get_metric(ssimu2_json_path, "SSIMULACRA2")
                xpsnr_json_path = self.temp_dir / f"{self.input_path.stem}_xpsnr.json"
                xpsnr_scores, _ = self.get_metric(xpsnr_json_path, "XPSNR")

                if method_name == 'minimum':
                    ssimu2_average = mean(ssimu2_scores)

                ssimu2_scores_expanded = []
                for score in ssimu2_scores:
                    ssimu2_scores_expanded += [score] + [-1] * (skip - 1)

                xpsnr_scores_expanded = []
                for score in xpsnr_scores:
                    xpsnr_scores_expanded += [score] + [-1] * (skip - 1)

                total_scores = []
                self.percentile_5_total = []
                for i in range(len(self.ranges) - 1):
                    chunk_ssimu2_scores_expanded = ssimu2_scores_expanded[
                        self.ranges[i]:self.ranges[i+1]
                        ]
                    chunk_xpsnr_scores_expanded = xpsnr_scores_expanded[
                        self.ranges[i]:self.ranges[i+1]
                        ]

                    chunk_ssimu2_scores = [
                        score for score in chunk_ssimu2_scores_expanded if score != -1
                        ]
                    chunk_xpsnr_scores = [
                        score for score in chunk_xpsnr_scores_expanded if score != -1
                        ]
                    chunk_scores = []
                    for i, ssimu2_score in enumerate(chunk_ssimu2_scores):
                        xpsnr_score = chunk_xpsnr_scores[i]

                        if method_name == "multiplied":
                            chunk_scores.append(ssimu2_score * xpsnr_score)
                        elif method_name == "minimum":
                            chunk_scores.append(
                                min(ssimu2_score, ssimu2_average * xpsnr_score) # type: ignore
                                )

                    total_scores += chunk_scores
                    chunk_percentile_5 = quantiles(chunk_scores, n=100)[4]
                    self.percentile_5_total.append(chunk_percentile_5)

                self.average = mean(total_scores)

                # print(f'Minimum:')
                # print(f'Median score:  {calculation_average}')
                # print(f'5th Percentile:  {calculation_percentile_5}')
                # print(f'95th Percentile:  {calculation_percentile_95}\n')
                self.generate_zones()


    def generate_zones(self) -> None:
        """
        Appends a scene change to the ``zones_txt_path`` file in Av1an zones format.

        creates ``zones_txt_path`` if it does not exist. If it does exist, the line is
        appended to the end of the file.

        :param ranges: Scene changes list
        :type ranges: list
        :param percentile_5_total: List containing all 5th percentile scores
        :type percentile_5_total: list
        :param average: Full clip average score
        :type average: int
        :param crf: CRF setting to use for the zone
        :type crf: int
        :param zones_txt_path: Path to the zones.txt file
        :type zones_txt_path: str
        :param video_params: custom encoder params for av1an
        :type video_params: str
        """
        zones_iter = 0

        # If only one is set, use base deviation as the other limit
        if self.max_positive_deviation is None:
            self.max_positive_deviation = self.base_deviation
        if self.max_negative_deviation is None:
            self.max_negative_deviation = self.base_deviation

        for i in range(len(self.ranges)-1):
            zones_iter += 1

            # Calculate CRF adjustment using aggressive or normal multiplier
            multiplier = self.aggressiveness
            adjustment = ceil(
                (1.0 - (self.percentile_5_total[i] / self.average)) * multiplier * 4) / 4
            new_crf = self.crf - adjustment

            # Calculation explanation
            # percentile5/avg give a quality % of the frame compared to average.
            # 1 = average (Good); <1 = lower than average (bad, needs up boost);
            # >1 = higher than average (~bad, can be reduced)
            # 1 - percentile5/avg inverts the previous ratio so the closer to zero,
            # the better the frame, the further to zero the worst the frame
            # * multiplier highlights the extremes so they get more boosted,
            # the higher the multiplier, the more the frame is boosted
            # ceil(... * 4 ) / 4 round the adjustment to 0.25

            # Apply deviation limits
            if adjustment < 0:  # Positive deviation (increasing CRF)
                if self.max_positive_deviation == 0:
                    new_crf = self.crf  # Never increase CRF if max_pos_dev is 0
                elif abs(adjustment) > self.max_positive_deviation:
                    new_crf = self.crf + self.max_positive_deviation
            else:  # Negative deviation (decreasing CRF)
                if self.max_negative_deviation == 0:
                    new_crf = self.crf  # Never decrease CRF if max_neg_dev is 0
                elif abs(adjustment) > self.max_negative_deviation:
                    new_crf = self.crf - self.max_negative_deviation

            # print(f'Enc:  [{ranges[i]}:{ranges[i+1]}]\n'
            #       f'Chunk 5th percentile: {percentile_5_total[i]}\n'
            #       f'CRF adjustment: {adjustment:.2f}\n'
            #       f'Final CRF: {new_crf:.2f}\n')

            zone_params = f"--crf {new_crf:.2f}"

            if self.workers > 12:
                zone_params += " --lp 2"

            if self.video_parameters:  # Only append video_params if it exists and is not None
                zone_params += f' {self.video_parameters}'

            with open(self.zones_path, "w" if zones_iter == 1 else "a", encoding="utf-8") as file:
                file.write(f"{self.ranges[i]} {self.ranges[i+1]} svt-av1 {zone_params}\n")

        print(f"Auto-Boost complete\nSee '{self.zones_path}'")

    def get_metric(self, json_path: Path, metric: str) -> tuple[list[float], int]:
        """Reads from a json file and returns the metric scores and skip values"""
        scores: list[float] = []

        with open(json_path, "r", encoding="utf-8") as file:
            content = json.load(file)
            skip: int = content["skip"]
            scores = content[metric]

        return scores, skip

def parse_args() -> argparse.Namespace:
    """Argument parser function, returns parsed args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Video input filepath (original source file)")
    parser.add_argument("-t", "--temp",
                        help="The temporary directory for av1an to store files in"
                             " (Default: input filename)")
    parser.add_argument("-s", "--stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Select stage: 0: All, 1: fastpass, 2: calculate metrics,"
                             " 3: generate zones, 4: finalpass (Default = 0)")
    parser.add_argument("-crf", type=float, default=30.0,
                        help="Base CRF (Default: 30.0)")
    parser.add_argument("-d", "--deviation", type=float, default=10.0,
                        help="Base deviation limit for CRF changes "
                             "(used if max_positive_dev or max_negative_dev not set)"
                             " (Default: 10.0)")
    parser.add_argument("--max-positive-dev", type=float, default=None,
                        help="Maximum allowed positive CRF deviation (Default: None)")
    parser.add_argument("--max-negative-dev", type=float, default=None,
                        help="Maximum allowed negative CRF deviation (Default: None)")
    parser.add_argument("-p", "--preset", type=int,
                        help="Encoding preset")
    parser.add_argument("-p1", "--fast-preset", type=int,
                        help="Fast pass preset (Default: 6)")
    parser.add_argument("-p2", "--final-preset", type=int,
                        help="Final pass preset (Default: 4)")
    parser.add_argument("-w", "--workers", type=int, default=psutil.cpu_count(logical=False),
                        help="Number of av1an workers (Default: Number of physical cpu cores)")
    parser.add_argument("-S", "--skip", type=int,
                        help="Skip value, the metric is calculated every nth frames (Default: 1)")
    parser.add_argument("--ssimulacra2-skip", type=int,
                        help="SSIMULACRA2 Skip value")
    parser.add_argument("--xpsnr-skip", type=int,
                        help="XPSNR skip value")
    parser.add_argument("-m", "--method",type=int, default=3, choices=[1, 2, 3, 4],
    parser.add_argument("-m", "--method",type=int, default=3, choices=[1, 2, 3, 4],
                        help="Zones calculation method: 1 = SSIMU2, 2 = XPSNR,"
                        " 3 = Multiplication, 4 = Lowest Result (Default: 1)")
    parser.add_argument("-a", "--aggressiveness", type=float, default=20.0,
                        help="Choose aggressiveness, use 40 for more aggressive (Default: 20.0)")
    parser.add_argument("-M","--metrics-implementations", default="vship,vszip", type=str,
                        help="Metrics calculation implementation SSIMULACRA2,XPSNR"
                        " (Default: vship,vszip)")
    parser.add_argument("-v","--video-params", default="",
                        help="Custom encoder parameters for av1an")
    parser.add_argument("-ef", "--encoder-framework", choices=["av1an", "builtin"],
                        default="av1an",
                        help="Choose encoder framework from av1an or builtin"
                        " (Not implemented yet) (Default: av1an)")
    parser.add_argument("-o", "--output",
                        help="Output file path for final encode (Default: <input>_boosted.mkv)")
    # parser.add_argument("--verbose", action="store_true",
    #                     help="Enables a more verbose console output (Default: not active)")
    return parser.parse_args()

def main():
    """Main function, managing stages"""
    args = parse_args()

    # Directories / Files
    input_path: Path = Path(args.input).resolve()

    output_path: Optional[Path] = Path(args.output) if args.output else None
    temp_dir: Optional[Path] = Path(args.temp) if args.temp else None

    # Computation Parameters
    stage: int = args.stage
    method: int = args.method
    base_deviation: float = args.deviation
    max_positive_deviation: Optional[float] = args.max_positive_dev
    max_negative_deviation: Optional[float] = args.max_negative_dev
    aggressiveness: float = args.aggressiveness

    skip = args.skip if args.skip else None
    ssimulacra2_skip = args.ssimulacra2_skip if args.ssimulacra2_skip else None
    xpsnr_skip = args.xpsnr_skip if args.xpsnr_skip else None

    # Encoding Parameters
    preset = args.preset if args.preset else None
    fastpass_preset = args.fast_preset if args.fast_preset else None
    finalpass_preset = args.final_preset if args.final_preset else None

    crf: float = args.crf
    video_parameters: str = args.video_params
    metrics_implementations = args.metrics_implementations

    # encoder_framework: str = args.encoder_framework
    workers: int = args.workers

    autoboost = AutoBoost(
        input_path,
        output_path,
        temp_dir,
        workers,
        video_parameters,
        preset,
        fastpass_preset,
        finalpass_preset,
        crf,
        method,
        metrics_implementations,
        skip,
        ssimulacra2_skip,
        xpsnr_skip,
        base_deviation,
        max_positive_deviation,
        max_negative_deviation,
        aggressiveness,
        )

    match stage:
        case 0:
            autoboost.run_all()
        case 1:
            autoboost.fastpass()
        case 2:
            autoboost.measure_metrics()
        case 3:
            autoboost.boost()
        case 4:
            autoboost.finalpass()
    return None


if __name__ == '__main__':
    main()
