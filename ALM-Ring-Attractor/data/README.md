# Data

## Download

Raw Neuropixels data are not included in this repository due to file size.

Download all NWB files from:  
**DANDI Archive 000060**  
https://dandiarchive.org/dandiset/000060

## Setup

1. Create subfolder: `data/raw/` on your local machine
2. Download `.nwb` files from DANDI into `data/raw/`
3. Run analysis: `cd ../code && python finaloutput2DRing.py`

The script expects NWB files at: `data/raw/*.nwb`

## Note
The `raw/` folder is in `.gitignore` and will not be uploaded to GitHub.