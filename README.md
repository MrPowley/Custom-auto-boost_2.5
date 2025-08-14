# Custom Auto-Boost_2.5
[![lang - EN](https://img.shields.io/badge/lang-EN-d5372d?style=for-the-badge)](README.md)
[![lang - FR](https://img.shields.io/badge/lang-FR-2d3181?style=for-the-badge)](README.fr.md)

This project is a modified version of [nekotrix](https://github.com/nekotrix) project [auto-boost-algorithm](https://github.com/nekotrix/auto-boost-algorithm) version 2.5.

## Version 2.5: SSIMULACRA2&XPSNR-based

Requirements:
- [Vapoursynth](https://github.com/vapoursynth/vapoursynth)
- LSMASHSource
- [Av1an](https://github.com/rust-av/Av1an/)
- [vszip](https://github.com/dnjulek/vapoursynth-zip)
- tqdm
- psutil
- [mkvmerge](https://www.matroska.org/index.html)
  
Optionally: 
- ffmpeg built with XPSNR support
- [turbo-metrics](https://github.com/Gui-Yom/turbo-metrics)

# Notable changes from original
- Added a progressbar to the turbo-metrics calculation
- Added a stage 4 for a final encode with Av1an
- (Technically more accurate by a small margin)

# Todo, functionality
- (WIP) Add a home-made encoding framework (alternative to Av1an), because why not
- Be able to adjust aggressiveness more precisely
- Remove the ffmpeg dependency by using vs-zip xpsnr implementation
- Improve CLI
- (Maybe) Add a TUI
- (Some day) Add a GUI

# Todo, code
- Make/Keep the code fully [pep8](https://peps.python.org/pep-0008/) compliant
- Refactor the code to have a Controller-Model-View architecture

# Usage
|Parameter|Default|Possible values|Explanation|
|---|---|---|---|
|`-i, --input`|||Video input file path|
|`-t, --temp`|Input filename||Temporary directory|
|`-s, --stage`|`0`|`0`, `1`, `2`, `3`, `4`|Boosting stage (All, fast pass, metrics calculation, zones generation, final encode)|
|`-crf`|`30`|`1`-`63`|CRF to use|
|`-d, -deviation`|`10`|`0`-`63`|Maximum crf deviation from base crf|
|`--max-positive-dev`||`0`-`63`|Maximum positive deviation from base crf|
|`--max-negative-dev`||`0`-`63`|Maximum negative deviation from base crf|
|`-p, --preset`|`8`|`-1`-`13`|SVT-AV1 preset to use|
|`-w, --workers`|CPU core count| `1`-`Any`|Number of Av1an workers|
|`-S, --skip`|`1` (GPU), `3` (CPU)|`1`-`Any`|Calculate metrics every X frames|
|`-m, --method`|`1`|`1`, `2`, `3`, `4`|Zones calculation method (1: SSIMU2, 2: XPSNR, 3: Multiplied, 4: Minimum)|
|`-a, --aggressiveness`|`20` or `40`|`0` - `Any`|Choose boosting aggressiveness|
|`-cpu, --force-cpu`|Not active||Force the use of CPU for SSIMU2 calculation|
|`-gpu, --hwaccel`|`turbo-metrics`|`turbo-metrics`, `vship`|SSIMU2 hwaccel calculation framework|
|`-v --video-params`|||Encoder parameters for Av1an|
|`-ef, --encoder-framework`|`av1an`|`av1an`|Encoding framework to use|
|`-o, --output`|Input file directory||Output file for final encode|

---
_This project is based on the original work by **nekotrix**._  
_Original contributors include **R1chterScale**, **Yiss**, **Kosaka**, and others._  
_Licensed under MIT, see LICENSE._