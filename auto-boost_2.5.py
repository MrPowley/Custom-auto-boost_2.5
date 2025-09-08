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
import subprocess
import argparse

import psutil

import ui
import metrics
import encoders

def get_ranges(scenes: Path) -> list[int]:
    """
    Reads a scene file and returns a list of frame numbers for each scene change.

    Args:
        scenes (Path): scenes.json path

    Returns:
        List of ranges (int)
    """
    ranges = [0]

    with open(scenes, 'r', encoding='utf-8') as file:
        content = json.load(file)
        for scene in content['scenes']:
            ranges.append(scene['end_frame'])

    return ranges

def get_metric(json_path: Path, metric: str) -> tuple[list[float], int]:
    """Reads from a json file and returns the metric scores and skip values"""
    scores: list[float] = []

    with open(json_path, "r", encoding="utf-8") as file:
        content = json.load(file)
        skip: int = content["skip"]
        scores = content[metric]

    return scores, skip

def calculate_percentile(scores: list[float], percentile: int) -> float:
    """Inputs a sorted score list and output the 5th percentile"""
    if 0 >= percentile > 100:
        raise ValueError("Invalid 'percentile' value")

    scores_length = len(scores)

    fith_percent_index = percentile/100 * (scores_length + 1) - 1
    frac_part = fith_percent_index % 1.0
    lower_value = scores[int(fith_percent_index)]

    if frac_part == 0:
        percentile_value: float = lower_value
    else:
        upper_index = min(int(ceil(fith_percent_index)), scores_length - 1)
        upper_value = scores[upper_index]
        percentile_value: float = lower_value + frac_part * (upper_value - lower_value)

    return percentile_value

def calculate_std_dev(score_list: list[float]) -> tuple[float, float, float]:
    """
    Takes a list of metrics scores and returns the associated arithmetic mean,
    5th percentile and 95th percentile scores.

    :param score_list: list of SSIMU2 scores
    :type score_list: list
    """

    filtered_score_list = [score if score >= 0 else 0.0 for score in score_list]
    sorted_score_list = sorted(score_list)
    percentile_5 = calculate_percentile(sorted_score_list, 5)
    percentile_95 = calculate_percentile(sorted_score_list, 95)
    average = sum(filtered_score_list)/len(filtered_score_list)

    return average, percentile_5, percentile_95

def generate_zones(ranges: list[int], percentile_5_total: list[float], average: float,
                   crf: float, zones_txt_path: Path, video_params: str,
                   max_pos_dev: float | None, max_neg_dev: float | None,
                   base_deviation: float, aggressiveness: float, workers: int) -> None:
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
    if max_pos_dev is None:
        max_pos_dev = base_deviation
    if max_neg_dev is None:
        max_neg_dev = base_deviation

    for i in range(len(ranges)-1):
        zones_iter += 1

        # Calculate CRF adjustment using aggressive or normal multiplier
        multiplier = aggressiveness
        adjustment = ceil((1.0 - (percentile_5_total[i] / average)) * multiplier * 4) / 4
        new_crf = crf - adjustment

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
            if max_pos_dev == 0:
                new_crf = crf  # Never increase CRF if max_pos_dev is 0
            elif abs(adjustment) > max_pos_dev:
                new_crf = crf + max_pos_dev
        else:  # Negative deviation (decreasing CRF)
            if max_neg_dev == 0:
                new_crf = crf  # Never decrease CRF if max_neg_dev is 0
            elif abs(adjustment) > max_neg_dev:
                new_crf = crf - max_neg_dev

        # print(f'Enc:  [{ranges[i]}:{ranges[i+1]}]\n'
        #       f'Chunk 5th percentile: {percentile_5_total[i]}\n'
        #       f'CRF adjustment: {adjustment:.2f}\n'
        #       f'Final CRF: {new_crf:.2f}\n')

        zone_params = f"--crf {new_crf:.2f}"

        if workers > 12:
            zone_params += " --lp 2"

        if video_params:  # Only append video_params if it exists and is not None
            zone_params += f' {video_params}'

        with open(zones_txt_path, "w" if zones_iter == 1 else "a", encoding="utf-8") as file:
            file.write(f"{ranges[i]} {ranges[i+1]} svt-av1 {zone_params}\n")

    print(f"Auto-Boost complete\nSee '{zones_txt_path}'")
    return None

def calculate_xpsnr(
        source_file: Path, output_file: Path, json_path: Path, implementation: dict) -> None:
    """Handles implementation and calls the appropriated XPSNR calculation method"""
    implementation_handlers = {
        "vszip": [metrics.VSzipXPSNR],
        "ffmpeg": [metrics.FFmpegXPSNR, metrics.VSzipXPSNR]
    }

    for handler in implementation_handlers.get(implementation["implementation"]):
        print(f"Running {handler}")
        metric = init_metric(
            source_file, output_file, json_path, implementation["skip"], handler, "XPSNR")
        returncode = metric.run()
        if returncode == 0:
            return None
    raise RuntimeError("All XPSNR metric implementations failed")

def calculate_ssimulacra2(
        source_file: Path, output_file: Path, json_path: Path, implementation: dict) -> None:
    """Handles implementation and calls the appropriated SSIMULACRA2 calculation method"""

    implementation_handlers = {
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
    }

    for handler in implementation_handlers.get(implementation["implementation"]):
        metric = init_metric(
            source_file, output_file, json_path, implementation["skip"], handler, "SSIMULACRA2")
        returncode = metric.run()
        if returncode == 0:
            return None
    raise RuntimeError("All SSIMULACRA2 metric implementations failed")

def init_metric(
        source_file: Path, output_file: Path, json_path: Path,
        skip: int, metric_class, metric_name: str):
    """Initializes the metric and its progressbar"""
    progressbar = ui.ProgressBar()
    metric = metric_class(source_file, output_file, json_path, skip, progressbar.update_progressbar)

    video_length = metric.get_length()

    progressbar.initialize_progressbar(total=video_length, description=f"Calculating {metric_name}")

    return metric

def calculate_metrics(src_file: Path, output_file: Path, tmp_dir: Path,
                      method: int, implementation: dict) -> None:
    """Handles method arg to calculate appropriate metric"""
    ssimu2_json_path = tmp_dir / f"{src_file.stem}_ssimulacra2.json"
    xpsnr_json_path = tmp_dir / f"{src_file.stem}_xpsnr.json"

    match method:
        case 1:
            calculate_ssimulacra2(
                src_file, output_file, ssimu2_json_path, implementation["ssimulacra2"])
        case 2:
            calculate_xpsnr(src_file, output_file, xpsnr_json_path, implementation["xpsnr"])
        case 3|4:
            calculate_ssimulacra2(
                src_file, output_file, ssimu2_json_path, implementation["ssimulacra2"])
            calculate_xpsnr(src_file, output_file, xpsnr_json_path, implementation["xpsnr"])

def calculate_zones(
        src_file: Path, tmp_dir: Path, ranges: list[int],
        method: int, cq: float, video_params: str,
        max_pos_dev: float|None, max_neg_dev: float|None,
        base_deviation: float, aggressiveness: float, workers: int
        ) -> None:
    """Calcules zones with chosen method and metric"""
    match method:
        case 1 | 2:
            if method == 1:
                metric = 'SSIMULACRA2'
            else:
                metric = 'xpsnr'

            metric_json_path = tmp_dir / f'{src_file.stem}_{metric}.json'
            metric_scores, skip = get_metric(metric_json_path, metric)

            metric_zones_txt_path = tmp_dir / f'{metric}_zones.txt'

            # Expand the scores list with dummy values
            #  to the full length of the video to compensate the skip
            metric_scores_expanded = []
            for score in metric_scores:
                metric_scores_expanded += [score] + [-1] * (skip - 1)

            total_scores = []
            percentile_5_total = []
            for i in range(len(ranges) - 1):
                if metric_scores_expanded[ranges[i]:ranges[i+1]]:
                    chunk_scores_expanded = metric_scores_expanded[ranges[i]:ranges[i+1]]
                    # Remove the dummy values
                    chunk_scores = [score for score in chunk_scores_expanded if score != -1]
                    # print(chunk_scores, chunk_scores_expanded)
                    _, chunk_percentile_5, _ = calculate_std_dev(chunk_scores)
                    percentile_5_total.append(chunk_percentile_5)
                    total_scores += chunk_scores

            metric_average, _, _ = calculate_std_dev(total_scores)

            # print(f'{metric}')
            # print(f'Median score: {metric_average}')
            # print(f'5th Percentile: {metric_percentile_5}')
            # print(f'95th Percentile: {metric_percentile_95}')
            generate_zones(ranges, percentile_5_total, metric_average, cq,
                           metric_zones_txt_path, video_params, max_pos_dev, max_neg_dev,
                           base_deviation, aggressiveness, workers)

        case 3 | 4:
            if method == 3:
                method_name = 'multiplied'
            else:
                method_name = 'minimum'

            ssimu2_json_path = tmp_dir / f"{src_file.stem}_ssimulacra2.json"
            ssimu2_scores, skip = get_metric(ssimu2_json_path, "SSIMULACRA2")
            xpsnr_json_path = tmp_dir / f"{src_file.stem}_xpsnr.json"
            xpsnr_scores, _ = get_metric(xpsnr_json_path, "XPSNR")

            calculation_zones_txt_path = tmp_dir / f"{method_name}_zones.txt"

            if method_name == 'minimum':
                ssimu2_average, _, _ = calculate_std_dev(ssimu2_scores)

            ssimu2_scores_expanded = []
            for score in ssimu2_scores:
                ssimu2_scores_expanded += [score] + [-1] * (skip - 1)

            xpsnr_scores_expanded = []
            for score in xpsnr_scores:
                xpsnr_scores_expanded += [score] + [-1] * (skip - 1)

            total_scores = []
            percentile_5_total = []
            for i in range(len(ranges) - 1):
                chunk_ssimu2_scores_expanded = ssimu2_scores_expanded[ranges[i]:ranges[i+1]]
                chunk_xpsnr_scores_expanded = xpsnr_scores_expanded[ranges[i]:ranges[i+1]]

                chunk_ssimu2_scores = [
                    score for score in chunk_ssimu2_scores_expanded if score != -1
                    ]
                chunk_xpsnr_scores = [score for score in chunk_xpsnr_scores_expanded if score != -1]
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
                _, chunk_percentile_5, _ = calculate_std_dev(chunk_scores)
                percentile_5_total.append(chunk_percentile_5)

            calculation_average, _, _ = calculate_std_dev(total_scores)

            # print(f'Minimum:')
            # print(f'Median score:  {calculation_average}')
            # print(f'5th Percentile:  {calculation_percentile_5}')
            # print(f'95th Percentile:  {calculation_percentile_95}\n')
            generate_zones(ranges, percentile_5_total, calculation_average, cq,
                           calculation_zones_txt_path,video_params, max_pos_dev, max_neg_dev,
                           base_deviation,aggressiveness, workers)


def parse_args() -> argparse.Namespace:
    """Argument parser function, returns parsed args"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Video input filepath (original source file)"
                        )
    parser.add_argument("-t", "--temp",
                        help="The temporary directory for av1an to store files in"
                             " (Default: input filename)")
    parser.add_argument("-s", "--stage", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="Select stage: 0: All, 1: fastpass, 2: calculate metrics,"
                             " 3: generate zones (Default = 0)")
    parser.add_argument("-crf", type=float, default=30.0,
                        help="Base CRF (Default: 30.0)")
    parser.add_argument("-d", "--deviation", type=float, default=10.0,
                        help="Base deviation limit for CRF changes "
                             "(used if max_positive_dev or max_negative_dev not set)"
                             " (Default: 10.0)")
    parser.add_argument("--max-positive-dev", type=float, default=None,
                        help="Maximum allowed positive CRF deviation (Default: None)")
    parser.add_argument("--max-negative-dev", type=float, default=None,
                        help="Maximum allowed negative CRF deviation | Default: None")
    parser.add_argument("-p", "--preset", type=int,
                        help="Fast encode preset")
    parser.add_argument("-p1", "--fast-preset (Default: 7)", type=int, default=7,
                        help="Fast pass preset")
    parser.add_argument("-p2", "--final-preset (Default: 6)", type=int, default=6,
                        help="Final pass preset")
    parser.add_argument("-w", "--workers", type=int, default=psutil.cpu_count(logical=False),
                        help="Number of av1an workers (Default: Depends on physical cores number)")
    parser.add_argument("-S", "--skip", type=int, default=3,
                        help="Skip value, the metric is calculated every nth frames (Default: 3)")
    parser.add_argument("--ssimulacra2-skip", type=int,
                        help="SSIMULACRA2 Skip value")
    parser.add_argument("--xpsnr-skip", type=int,
                        help="XPSNR skip value")
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
                        help="Output file path for final encode (Default: input directory)")
    return parser.parse_args()

def resolve_implementation(
        string: str, method: int, ssimulacra2_skip: int, xpsnr_skip: int) -> dict:
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
            "implementation": implementations[ssimulacra2_index], "skip": ssimulacra2_skip_value},
        "xpsnr": {
            "implementation": implementations[xpsnr_index], "skip": xpsnr_skip_value}
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
    if implementations_dict["ssimulacra2"]["implementation"] == "vszip" and not ssimulacra2_skip:
        implementations_dict["ssimulacra2"]["skip"] = 3

    return implementations_dict

def main():
    """Main function, managing stages"""
    args = parse_args()

    # Directories / Files
    src_file: Path = Path(args.input)
    output_dir: Path = src_file.parent

    if args.temp is not None:
        tmp_dir: Path = Path(args.temp).resolve()
    else:
        tmp_dir: Path = output_dir / src_file.stem

    fastpass_temp_dir = tmp_dir / "fastpass"
    finalpass_temp_dir = tmp_dir / "finalpass"

    fastpass_path: Path = tmp_dir / "fastpass.mkv"
    scenes_path: Path = tmp_dir / "scenes.json"

    if args.output is not None:
        output_file: Path = args.output
    else:
        output_file: Path = output_dir / f"{src_file.stem}_boosted.mkv"

    # Computation Parameters
    stage: int = args.stage
    method: int = args.method
    base_deviation: float = args.deviation
    max_pos_dev: float|None = args.max_positive_dev
    max_neg_dev: float|None = args.max_negative_dev
    aggressiveness: float = args.aggressiveness

    ssimulacra2_skip = args.ssimulacra2_skip
    xpsnr_skip = args.xpsnr_skip

    if args.skip:
        ssimulacra2_skip = args.skip
        xpsnr_skip = args.skip

    metric_implementation = resolve_implementation(
        args.metrics_implementations, method, ssimulacra2_skip, xpsnr_skip)

    # Encoding Parameters
    fast_pass_preset = args.fast_preset
    final_pass_preset = args.final_preset

    if args.preset and not fast_pass_preset:
        fast_pass_preset = args.preset
    if args.preset and not final_pass_preset:
        final_pass_preset = args.preset

    crf: float = args.crf
    video_params: str = args.video_params

    # encoder_framework: str = args.encoder_framework

    metric = "ssimulacra2"
    match method:
        case 2:
            metric = "xpsnr"
        case 3:
            metric = "multiplied"
        case 4:
            metric = "minimum"

    zones_path = tmp_dir / f'{metric}_zones.txt'
    workers: int = args.workers

    if tmp_dir.exists() and stage == 1:
        user_cancel = input(
            "Temporary directory already exists and will be erased, continue ? [Y/n]"
            )
        if user_cancel.lower() not in ("y", ""):
            exit(1)

    if not src_file.is_file():
        raise FileNotFoundError(f"File: '{str(src_file)}' doesn't exist.")

    if not tmp_dir.exists or not tmp_dir.is_dir:
        raise NotADirectoryError(f"Directory: '{str(tmp_dir)}' doens't exist or is not a directory")

    match stage:
        case 0:
            av1an = encoders.Av1an(src_file, workers, video_params)
            av1an.fast_pass(fastpass_path, fastpass_temp_dir, scenes_path, fast_pass_preset, crf)

            ranges = get_ranges(scenes_path)

            calculate_metrics(src_file, fastpass_path, tmp_dir, method, metric_implementation)
            calculate_zones(src_file, tmp_dir, ranges, method, crf,
                            video_params, max_pos_dev, max_neg_dev,
                            base_deviation, aggressiveness, workers)

            av1an.final_pass(output_file, finalpass_temp_dir, zones_path, final_pass_preset)
        case 1:
            av1an = encoders.Av1an(src_file, workers, video_params)
            av1an.fast_pass(fastpass_path, fastpass_temp_dir, scenes_path, fast_pass_preset, crf)
        case 2:
            calculate_metrics(src_file, fastpass_path, tmp_dir, method, metric_implementation)
        case 3:
            ranges = get_ranges(scenes_path)
            calculate_zones(src_file, tmp_dir, ranges, method, crf,
                            video_params, max_pos_dev, max_neg_dev,
                            base_deviation, aggressiveness, workers)
        case 4:
            av1an = encoders.Av1an(src_file, workers, video_params)
            av1an.final_pass(output_file, finalpass_temp_dir, zones_path, final_pass_preset)
    return None


if __name__ == '__main__':
    main()
