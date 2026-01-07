# PRKCE-Golgi-Imaging
Automated Golgi morphology analysis tool. 
This repository contains a Python-based tool designed for quantifying Golgi apparatus morphology.

## Overview
This script automates the analysis of Golgi fragmentation and 2D spatial distribution using ImageJ ROI data. It was specifically developed to investigate Golgi integrity in relation to **PRKCE** expression.

## Key features
* **Two-Channel Analysis:** Processes multiple channels (e.g., Golgi markers and protein of interest) simultaneously.
* **Fragmentation Quantification:** Calculates particle counts and individual surface areas.
* **Convex Hull Measurement:** Determines the overall spatial extent of the Golgi apparatus.
* **Scale Integration:** Automatically applies physical units (µm²) based on scale bar detection.

## Requirements
* Python 3.x
* OpenCV, NumPy, Pandas, PyQt5
* `read_roi`, `roifile`

## Project Structure
* `Golgi_twochannel_PRKCE.py`: Main analysis script.
* `scalebar_detection_dist.py`: Module for automatic scale bar detection and calibration.

## Usage
1. Place your `.tif` images and corresponding `.zip` ROI files in the same folder.
2. Run the script: `python PRKCE_Golgi_twochannel.py`
3. Select the target folder when the dialog appears.
