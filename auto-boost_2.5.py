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
# import os
import subprocess
# import re
import argparse
# import shutil
# import platform

# from tqdm import tqdm
import psutil
import vapoursynth as vs

import ui
import metrics

# from encode import Encoder

# IS_WINDOWS = platform.system() == 'Windows'
# NULL_DEVICE = 'NUL' if IS_WINDOWS else '/dev/null'

core = vs.core
core.max_cache_size = 1024

def get_ranges(scenes: Path) -> list[int]:
    """
    Reads a scene file and returns a list of frame numbers for each scene change.

    Args:
        scenes (Path): scenes.json path

    Returns:
        List of ranges (int)
    """
    ranges = [0]

    with open(scenes, "r") as file:
        content = json.load(file)
        for scene in content['scenes']:
            ranges.append(scene['end_frame'])

    return ranges


def fast_pass(
        input_file: Path,
        output_file: Path,
        tmp_dir: Path,
        scenes_file: Path,
        preset: int,
        crf: float,
        workers: int,
        video_params: str
    ) -> None:
    """
    Fast av1an encoding

    Args:
        input_file (Path): Input video file path
        output_file (Path): Output video file path
        tmp_dir (Path): Temporary directory path
        scenes_file (Path): scenes.json file path
        preset (int): Encoder preset
        crf (float): Encoder CRF
        workers (int): Number of av1an workers
        video_params (str): Encoder video parameters

    Returns:
        None
    """

    encoder_params = f'--preset {preset} --crf {crf:.2f} --lp 2 --keyint 0 --scm 0' \
                      ' --fast-decode 1 --color-primaries 1' \
                      ' --transfer-characteristics 1 --matrix-coefficients 1'

    if video_params:  # Only append video_params if it exists and is not None
        encoder_params += f' {video_params}'

    fast_av1an_command = [
        'av1an',
        '-i', str(input_file),
        '--temp', str(tmp_dir),
        '-y',
        '--verbose',
        '--keep',
        '-m', 'lsmash',
        '-c', 'mkvmerge',
        '--min-scene-len', '24',
        '--scenes', str(scenes_file),
        '--sc-downscale-height', '720',
        '--set-thread-affinity', '2',
        '-e', 'svt-av1',
        '--force',
        '-v', encoder_params,
        '-w', str(workers),
        '-o', str(output_file)
    ]

    try:
        subprocess.run(fast_av1an_command, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Av1an encountered an error:\n{e}")
        exit(1)

def get_metric(json_path: Path, metric: str) -> tuple[list[float], int]:
    """Reads from a json file and returns the metric scores and skip values"""
    scores: list[float] = []

    with open(json_path, "r") as file:
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
                   base_deviation: float, aggressiveness: float, workers: int):
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
        # percentile5/avg give a quality % of the frame compared to average. 1 = average (Good); <1 = lower than average (bad, needs up boost); >1 = higher than average (~bad, can be reduced)
        # 1 - percentile5/avg inverts the previous ratio so the closer to zero, the better the frame, the further to zero the worst the frame
        # * multiplier highlights the extremes so they get more boosted, the higher the multiplier, the more the frame is boosted
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

        with open(zones_txt_path, "w" if zones_iter == 1 else "a") as file:
            file.write(f"{ranges[i]} {ranges[i+1]} svt-av1 {zone_params}\n")

    print(f"Auto-Boost complete\nSee '{zones_txt_path}'")
    return None

def calculate_xpsnr(source_file: Path, output_file: Path, json_path: Path, implementation: dict):
    if implementation["implementation"].startswith("vs"):
        vs_metrics = init_metric(source_file, output_file, json_path, implementation["skip"], metrics.VS_metrics, metric_name="xpsnr")
        vs_metrics.xpsnr_vszip()
    else:
        raise NotImplementedError("(WIP) Use vszip for now")

def calculate_ssimulacra2(source_file: Path, output_file: Path, json_path: Path, implementation: dict):
    print(implementation)
    if implementation["implementation"].startswith("vs"):
        vs_metrics = init_metric(source_file, output_file, json_path, implementation["skip"], metrics.VS_metrics, metric_name="ssimulacra")

        if implementation["implementation"] == "vszip":
            vs_metrics.ssimulacra2_vszip()
        elif implementation["implementation"] == "vship":
            vs_metrics.ssimulcara2_vship()
    elif implementation["implementation"] == "turbo-metrics":
        turbo_metrics = init_metric(source_file, output_file, json_path, implementation["skip"], metrics.Turbo_metrics, metric_name="ssimulacra2")
        turbo_metrics.run("ssimulacra2")
    else:
        raise NotImplementedError("(WIP) Use vszip or vship for now")

def init_metric(source_file: Path, output_file: Path, json_path: Path, skip: int, metric_class, metric_name: str):
    progressbar = ui.ProgressBar()
    metric = metric_class(source_file, output_file, json_path, skip, progressbar.update_progressbar)

    video_length = metric.get_length()

    progressbar.initialize_progressbar(total=video_length, description=f"Calculating {metric_name}")

    return metric

def calculate_metrics(src_file: Path, output_file: Path, tmp_dir: Path,
                      method: int, implementation: dict):
    ssimu2_json_path = tmp_dir / f"{src_file.stem}_ssimulacra2.json"
    xpsnr_json_path = tmp_dir / f"{src_file.stem}_xpsnr.json"

    match method:
        case 1:
            calculate_ssimulacra2(src_file, output_file, ssimu2_json_path, implementation["ssimulacra2"])
        case 2:
            calculate_xpsnr(src_file, output_file, xpsnr_json_path, implementation["xpsnr"])
        case 3|4:
            calculate_ssimulacra2(src_file, output_file, ssimu2_json_path, implementation["ssimulacra2"])
            calculate_xpsnr(src_file, output_file, xpsnr_json_path, implementation["xpsnr"])

def calculate_zones(src_file: Path, tmp_dir: Path, ranges: list[int],
                    method: int, cq: float, video_params: str,
                    max_pos_dev: float|None, max_neg_dev: float|None,
                    base_deviation: float, aggressiveness: float, workers: int):
    match method:
        case 1 | 2:
            if method == 1:
                metric = 'ssimulacra2'
            else:
                metric = 'xpsnr'

            metric_json_path = tmp_dir / f'{src_file.stem}_{metric}.json'
            metric_scores, skip = get_metric(metric_json_path, metric)

            metric_zones_txt_path = tmp_dir / f'{metric}_zones.txt'

            # Expand the scores list with dummy values to the full length of the video to compensate the skip
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
                chunk_ssimu2_scores_expanded: list[float] = ssimu2_scores_expanded[ranges[i]:ranges[i+1]]
                chunk_xpsnr_scores_expanded: list[float] = xpsnr_scores_expanded[ranges[i]:ranges[i+1]]

                chunk_ssimu2_scores = [score for score in chunk_ssimu2_scores_expanded if score != -1]
                chunk_xpsnr_scores = [score for score in chunk_xpsnr_scores_expanded if score != -1]
                chunk_scores = []
                for i, ssimu2_score in enumerate(chunk_ssimu2_scores):
                    xpsnr_score = chunk_xpsnr_scores[i]

                    if method_name == "multiplied":
                        chunk_scores.append(ssimu2_score * xpsnr_score)
                    elif method_name == "minimum":
                        chunk_scores.append(min(ssimu2_score, ssimu2_average * xpsnr_score)) # type: ignore

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
    parser.add_argument("-p", "--preset", type=int, default=8,
                        help="Fast encode preset (Default: 8)")
    parser.add_argument("-w", "--workers", type=int, default=psutil.cpu_count(logical=False),
                        help="Number of av1an workers (Default: Depends on physical cores number)")
    # parser.add_argument("-S", "--skip", type=int,
    #                     help="Skip value, the metric is calculated every nth frames"
    #                     " (Default: 1 for turbo-metrics/vship, 3 for vs-zip)")
    parser.add_argument("-m", "--method",type=int, default=1, choices=[1, 2, 3, 4],
                        help="Zones calculation method: 1 = SSIMU2, 2 = XPSNR,"
                        " 3 = Multiplication, 4 = Lowest Result (Default: 1)")
    parser.add_argument("-a", "--aggressiveness", type=float, default=20.0,
                        help="Choose aggressiveness, use 40 for more aggressive (Default: 20.0)")
    parser.add_argument("-M","--metrics-implementations", default="vszip,vszip",
                        help="Metrics calculation implementation (Default: vszip,vszip)")
    parser.add_argument("-v","--video-params", default="",
                        help="Custom encoder parameters for av1an")
    parser.add_argument("-ef", "--encoder-framework", choices=["av1an", "builtin"], default="av1an",
                        help="Choose encoder framework from av1an or builtin (Not implemented yet) (Default: av1an)")
    parser.add_argument("-o", "--output",
                        help="Output file path for final encode (Default: input directory)")
    return parser.parse_args()

def encode_av1an(src_file: Path, output_file: Path, zones_file: Path, video_params: str, workers: int):
    av1an_cmd = ["av1an", "-i", str(src_file), "-y", "--split-method", "none", "--verbose", "-e", "svt-av1", "-v", video_params.strip(), "--zones", str(zones_file), "-o", str(output_file), "-w", str(workers)]
    subprocess.run(av1an_cmd)

def resolve_implementation(string: str) -> dict:
    implementations = string.split(",")
    if len(implementations) < 2:
        implementations += [""]

    implementations_dict = {
        "ssimulacra2": {"implementation": implementations[0], "skip": 1},
        "xpsnr": {"implementation": implementations[1], "skip": 1}
    }

    if (
        not implementations_dict["ssimulacra2"]["implementation"]
         or implementations_dict["ssimulacra2"]["implementation"] not in ("vszip", "vship", "turbo-metrics")
        ):
        implementations_dict["ssimulacra2"]["implementation"] = "vszip"
        
    if (
        not implementations_dict["xpsnr"]["implementation"]
         or implementations_dict["xpsnr"]["implementation"] not in ("vszip")
        ):
        implementations_dict["xpsnr"]["implementation"] = "vszip"

    if implementations_dict["ssimulacra2"]["implementation"] == "vszip":
        implementations_dict["ssimulacra2"]["skip"] = 3

    return implementations_dict

def main():
    """Main function, managing stages"""
    args = parse_args()

    # if shutil.which("av1an") is None:
    #     raise FileNotFoundError("'av1an' is required but was not found in your system PATH. Install it and try again.")

    # Directories / Files
    src_file: Path = Path(args.input).resolve()
    output_dir: Path = src_file.parent
    tmp_dir: Path = Path(args.temp).resolve() if args.temp is not None else output_dir / src_file.stem
    fastpass_file: Path = tmp_dir / "fastpass.mkv"
    scenes_file: Path = tmp_dir / "scenes.json"
    output_file: Path = args.output if args.output is not None else output_dir / f"{src_file.stem}_boosted.mkv"

    # Computation Parameters
    stage: int = args.stage
    method: int = args.method
    base_deviation: float = args.deviation
    max_pos_dev: float|None = args.max_positive_dev
    max_neg_dev: float|None = args.max_negative_dev
    aggressiveness: float = args.aggressiveness

    metric_implementation = resolve_implementation(args.metric_implementation)

    # Encoding Parameters
    crf: float = args.crf
    preset: int = args.preset
    video_params: str = args.video_params

    encoder_framework: str = args.encoder_framework

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

    if not src_file.is_file():
        raise FileNotFoundError(f"File: '{str(src_file)}' doesn't exist.")

    if not tmp_dir.exists or not tmp_dir.is_dir:
        raise NotADirectoryError(f"Directory: '{str(tmp_dir)}' doens't exist or is not a directory")

    match stage:
        case 0:
            fast_pass(src_file, fastpass_file, tmp_dir, scenes_file, preset, crf, workers, video_params)
            ranges = get_ranges(scenes_file)
            calculate_metrics(src_file, fastpass_file, tmp_dir, method, metric_implementation)
            calculate_zones(src_file, tmp_dir, ranges, method, crf, video_params, max_pos_dev, max_neg_dev, base_deviation, aggressiveness, workers)
            encode_av1an(src_file, output_file, zones_path, f"--preset {preset} --lp 2 {" ".join(video_params.split())}",
                         workers)
        case 1:
            fast_pass(src_file, fastpass_file, tmp_dir, scenes_file, preset, crf, workers, video_params)
        case 2:
            calculate_metrics(src_file, fastpass_file, tmp_dir, method, metric_implementation)
        case 3:
            ranges = get_ranges(scenes_file)
            calculate_zones(src_file, tmp_dir, ranges, method, crf, video_params, max_pos_dev, max_neg_dev, base_deviation, aggressiveness, workers)
        case 4:
            encode_av1an(src_file, output_file, zones_path,
                             f"--preset {preset} --lp 2 {" ".join(video_params.split())}", workers)
    return None


if __name__ == '__main__':
    main()
