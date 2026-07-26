```text
=====================================================================================
 University of Study of Milano-Bicocca, Italy (MI)
 Dipartimento di Informatica, Sistemistica e Comunicazione (DISCo)
 Master Degree in Data Science - Digital Signal and Image Management
-------------------------------------------------------------------------------------
 Authors: Gebremichael Natnael Solomon (ID: 925350), Wani Godwill (ID: 925318), & Hussien Mohamed (ID: 925257)
=====================================================================================

 TASK 2  -  TWO-DIMENSIONAL SIGNALS (SEGMENTATION)
 Medical Image Segmentation via Custom U-Net
 Link to Google Drive folder: https://drive.google.com/drive/folders/1NhIYvdqrE_AASnmnlOAW7Ob_nzm8nC99?usp=drive_link

=====================================================================================
 0. OVERVIEW
=====================================================================================

This task tackles a computer vision problem on 2D endoscopic images: given a raw
RGB image from the Kvasir-SEG dataset, the goal is to localize polyps using
pixel-wise binary segmentation (Custom U-Net). 

=====================================================================================
 1. HOW TO READ AND REPRODUCE THE PROJECT
=====================================================================================

Run the Colab notebook cells sequentially:
    1. Environment Setup & Data Acquisition
    2. U-Net Segmentation (Architecture, Loss, Training, Evaluation)

=====================================================================================
 2. FOLDER AND FILE STRUCTURE
=====================================================================================

/content/
|-- drive/MyDrive/Digital Signal/Kvasir.zip        <-- Source dataset
|
|-- data/
    |-- processed/
        |-- unet/
        |   |-- images/ & masks/ (train/val/test)  <-- Stratified U-Net data
        |   |-- best_model.keras                   <-- Serialized U-Net
        |   |-- training_curves.png                <-- U-Net evaluation dashboard

=====================================================================================
 3. PIPELINE DESCRIPTION: U-NET SEGMENTATION
=====================================================================================

[Data Acquisition & Preprocessing]
- Extracted locally to bypass slow Google Drive I/O.
- Resized to 256x256, normalized to [0, 1].
- Masks binarized (pixels > 127 = 1.0, else 0.0).

[Architecture]
- Encoder: 4 blocks of Double Conv2D -> BatchNorm -> ReLU -> MaxPooling.
- Decoder: Conv2DTranspose + Skip Connections.
- Output: 1x1 Conv with Sigmoid for pixel-wise probability.

[Training]
- Loss: Custom `bce_dice_loss`.
- Metrics: Dice Coefficient and Intersection over Union (IoU).
=====================================================================================
```