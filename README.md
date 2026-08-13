# 2026 INTREN
A project aim to process DAS (Distributed Acoustic Sensing, 分散式光纖聲學感測) data. Specially designed for MiDAS data.

## Features
- Read MiniSEED data
- Wave and waterfall plot
- GIF images of waterfall and wave

## Installation
### pip
```bash
pip install -e .
```

### uv
```bash
uv sync
```

## Usage


## Project Structure
```bash
.
├── src/
│   ├── load/   # Load MiniSEED or CSV data
│   ├── plot/   # Generate figure
│   └── utils/  # Other tools
├── main.py     # Directly run the program
└── __main__.py # Allow users to use CLI interface
```

## Major Logs
- 2026/07/28, Search and wave cut feature
- 2026/07/29, Waterfall plot feature
- 2026/08/04, Use [dascore](https://github.com/DASDAE/dascore) to replace [obspy](https://github.com/obspy/obspy), inspired by [ralin3233](https://github.com/ralin3233/das-processing-pipeline)
- 2026/08/09, GIF feature

## Git Graph
```mermaid
%%{
    init: {
        'theme': 'neutral',
        'themeVariables': {
            'commitLabelFontSize': '18px'
        }
    }
}%%
gitGraph TB:
    commit id: "898f5fe init commit"
    commit id: "5f0b573 ready to coding"
    commit id: "d4afe4b add object Search that can select numerous midas files"
    commit id: "5ddb504 Add single search to Search"
    commit id: "6d018b0 add finger.py that can cut certain peroid of wave"
    commit id: "68b94ad Add waterfall.py that can generate waterfall plot"
    commit id: "70db0e6 bug fixed"
    commit id: "ff74c47 now they can save images"
    commit id: "57c4495 add typing"
    commit id: "1384ccb fix bugs and add load_data function"
    commit id: "1f79aec split Water in to Water and Fall"
    commit id: "b286e24 finally we got here"
    commit id: "2ce7893 fixed bugs"
    commit id: "e745b32 fix formate"
    commit id: "6ba8c15 fix X axis"
    commit id: "b9b970b change input"
    commit id: "911a6e1 Add uv"
    commit id: "8755c5f rebuild project"
    commit id: "410d27a change color bar"
    commit id: "4aa3eb7 use dascore to replace obspy"
    commit id: "a109678 rebuild project"
    commit id: "f334d87 Add useful tools"
    commit id: "193b3b9 add dascore"
    commit id: "15bd96b README update"
    commit id: "620a494 bug fixed"
    commit id: "0d76ad5 edit plot"
    commit id: "5cb7395 add tqdm, close Porgress Bar Fixes #1"
    commit id: "b115823 edit plot"
    commit id: "a3e817a add mutiple figures and figsize"
    commit id: "3ac6a0e plot fixed"
    commit id: "4d48ecb add random mkdir feature and time range convertor"
    commit id: "7f8d30c Add annotation and adjust figure layout"
    commit id: "82001f4 fix logos"
    commit id: "9b0491a add gif feature"
    commit id: "d92276f add pillow"
    commit id: "16794d7 add gif feature"
    commit id: "c173361 README update"
    commit id: "15864f4 add tqdm"
    commit id: "3ed0b93 Format change" tag: "HEAD, master"
```
