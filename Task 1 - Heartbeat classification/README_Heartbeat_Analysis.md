```text
=====================================================================================
 University of Study of Milano-Bicocca, Italy (MI)
 Dipartimento di Informatica, Sistemistica e Comunicazione (DISCo)
 Master Degree in Data Science - Digital Signal and Image Management
-------------------------------------------------------------------------------------
 Authors: Gebremichael Natnael Solomon (ID: 925350), Wani Godwill (ID: 925318), & Hussien Mohamed (ID: 925257)
=====================================================================================

 TASK 1  -  ONE-DIMENSIONAL SIGNALS (HEARTBEAT ANALYSIS)
 Cardiac Health Classification via DSP & Machine Learning
 Link to Google Drive folder: https://drive.google.com/drive/folders/1NhIYvdqrE_AASnmnlOAW7Ob_nzm8nC99?usp=drive_link

=====================================================================================
 0. OVERVIEW
=====================================================================================

This project focuses on mono-dimensional signal processing of audio signals for
cardiac health classification. Given a heartbeat sound recording (.wav), the goal 
is to classify it as healthy or unhealthy using both traditional machine learning 
(Random Forest) and deep learning (CNN) approaches.

=====================================================================================
 1. HOW TO READ AND REPRODUCE THE PROJECT
=====================================================================================

In order to reproduce the results from scratch, execute the Jupyter Notebook:
    1. Open `Heartbeat_Classification.ipynb` in your environment.
    2. Ensure dependencies are installed (librosa, scikit-learn, tensorflow, etc.)
    3. Verify the audio files are in the correct `data/` structure.
    4. Execute the cells sequentially to reproduce the analysis.

=====================================================================================
 2. FOLDER AND FILE STRUCTURE
=====================================================================================

scripts/
|-- Heartbeat_Classification.ipynb                     <-- Main analysis notebook
|
|-- data/                           <-- Dataset directory
    |-- train/                      <-- 3,240 audio files
    |   |-- healthy/
    |   |-- unhealthy/
    |-- val/                        <-- 301 audio files
        |-- healthy/
        |-- unhealthy/

=====================================================================================
 3. PIPELINE DESCRIPTION: HEARTBEAT ANALYSIS
=====================================================================================

[Signal Preprocessing]
- Bandpass filtering: 20-400 Hz range to focus on cardiac frequencies.
- Peak detection: Identifying individual heartbeat cycles.
- Beat segmentation: Extracting 0.3-second windows around detected peaks.

[Feature Extraction]
- Traditional ML: 13 MFCCs, Zero-crossing rate, Spectral centroid, Chroma.
- Deep Learning: 40-coeff MFCC 2D images, multi-feature maps (54x50x1).
- Normalization: Min-max scaling for consistent input ranges.

[Classification Models & Results]
- Random Forest: 100 estimators, 15-dimensional feature vector.
  -> Result: 100% Accuracy on validation set.
- Convolutional Neural Network (CNN): 3 Conv2D layers + MaxPool + Dropout.
  -> Result: ~50-68% validation accuracy (impacted by class imbalance).
=====================================================================================
```