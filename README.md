
### project over view

This repository contains the final coursework for the **Digital Signal and Image Management** course. The project demonstrates the application of Digital Signal Processing (DSP), Machine Learning, and Deep Learning techniques to solve real-world biomedical signal and medical imaging problems.

The project is organized into three independent tasks covering one-dimensional and two-dimensional biomedical data analysis.

# Task 1 – Heartbeat Classification

### Objective

Develop machine learning and deep learning models to classify heart sound recordings into **healthy** and **unhealthy** classes.

### Methods

- Digital Signal Processing
- Band-pass filtering
- Peak detection
- Beat segmentation
- MFCC feature extraction
- Random Forest
- Convolutional Neural Network (CNN)

### Dataset

Heartbeat audio recordings (.wav)

### Outcome

The task demonstrates how signal preprocessing and feature engineering improve automated cardiac health classification.

---

# Task 2 – Polyp Segmentation

### Objective

Automatically segment gastrointestinal polyps from endoscopic images using semantic segmentation.

### Methods

- Image preprocessing
- Binary mask generation
- Custom U-Net architecture
- Dice Loss
- Intersection over Union (IoU)

### Dataset

Kvasir-SEG Dataset


### Outcome

The developed U-Net accurately identifies the pixel-wise location of polyps, supporting computer-assisted diagnosis.

---

# Task 3 – Polyp Detection

### Objective

Detect and localize polyps in endoscopic images using an object detection framework.

### Methods

- YOLOv8 Nano
- Transfer Learning
- Data Augmentation
- Bounding Box Detection
- Performance Evaluation

### Dataset

Kvasir Dataset


### Outcome

The detection model predicts the location of polyps using bounding boxes, enabling fast and accurate lesion localization.

---

# Technologies Used

- Python
- TensorFlow / Keras
- Ultralytics YOLOv8
- Scikit-learn
- NumPy
- Pandas
- OpenCV
- Librosa
- Matplotlib
- Google Colab
- Jupyter Notebook

---

# Project Learning Outcomes

This project demonstrates practical experience in:

- Digital Signal Processing
- Medical Image Analysis
- Feature Engineering
- Machine Learning
- Deep Learning
- Convolutional Neural Networks
- U-Net Semantic Segmentation
- YOLO Object Detection
- Biomedical Data Analysis
- Performance Evaluation

---

# How to Run

Each task is self-contained.

Navigate to the corresponding task folder and follow the instructions provided in its `README.md`.

```
Task_1_Heartbeat_Classification/
Task_2_Polyp_Segmentation/
Task_3_Polyp_Detection/
```

Each folder contains:

- Project notebook
- Documentation
- Report
- Trained models (where applicable)
- Results and visualizations

---

# Acknowledgements

This project was completed as part of the **Digital Signal and Image Management** course in the Master's Degree in Data Science at the **University of Milano-Bicocca**.

---

## License

This repository is intended for educational and academic purposes.
