```text
=====================================================================================
 University of Study of Milano-Bicocca, Italy (MI)
 Dipartimento di Informatica, Sistemistica e Comunicazione (DISCo)
 Master Degree in Data Science - Digital Signal and Image Management
-------------------------------------------------------------------------------------
Authors: Gebremichael Natnael Solomon (ID: 925350), Wani Godwill (ID: 925318), & Hussien Mohamed (ID: 925257)
=====================================================================================

 TASK 3  -  TWO-DIMENSIONAL SIGNALS (DETECTION)
 Medical Image Detection via YOLOv8
 Link to Google Drive folder: https://drive.google.com/drive/folders/1NhIYvdqrE_AASnmnlOAW7Ob_nzm8nC99?usp=drive_link

=====================================================================================
 0. OVERVIEW
=====================================================================================

This task focuses on bounding-box object detection for polyps in endoscopic 
images using YOLOv8 Nano.

=====================================================================================
 1. HOW TO READ AND REPRODUCE THE PROJECT
=====================================================================================

Run the Colab notebook cells sequentially:
    1. Environment Setup & Data Acquisition
    2. YOLOv8 Detection (Pretrained weights, Training, Evaluation)

=====================================================================================
 2. FOLDER AND FILE STRUCTURE
=====================================================================================

/content/
|-- drive/MyDrive/Digital Signal/Kvasir.zip        <-- Source dataset
|
|-- data/
    |-- processed/
        |-- yolo/
        |   |-- kvasir.yaml                        <-- YOLO configuration
        |   |-- images/ & labels/ (train/val/test) <-- Stratified YOLO data
        |-- outputs/
            |-- yolo/train/weights/best.pt         <-- Serialized YOLOv8n

=====================================================================================
 3. PIPELINE DESCRIPTION: YOLOv8 DETECTION
=====================================================================================

[Architecture & Transfer Learning]
- Base Model: `yolov8n.pt` (Nano variant, 8.9 GFLOPs).
- Backbone: CSPDarknet.
- Neck: PANet.
- Head: Anchor-free decoupled head.

[Training & Augmentation]
- Configuration via `kvasir.yaml`.
- Augmentations: Mosaic (0.5), Horizontal Flip (0.5), Rotation (10 deg).
- Optimizer: AdamW.
=====================================================================================
```