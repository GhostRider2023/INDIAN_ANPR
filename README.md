# Indian Automatic Number Plate Recognition (ANPR) using EasyOCR and YOLOv8

This project uses **EasyOCR** and **YOLOv8** for Automatic Number Plate Recognition (ANPR) specifically for Indian car number plates. The system detects number plates from vehicles and extracts the registration numbers using Optical Character Recognition (OCR).

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Installation](#installation)
4. [Model Training](#model-training)
5. [Text Recognition with EasyOCR](#text-recognition-with-easyocr)
6. [Regular Expression Validation](#regular-expression-validation)
7. [Training Time Considerations](#training-time-considerations)
8. [Additional Considerations](#additional-considerations)
9. [Contributions](#contributions)

---

## Introduction

The goal of this project is to build an **Indian ANPR system** that detects and reads number plates on vehicles using **YOLOv8** for object detection and **EasyOCR** for text extraction. The system is trained specifically on the **Indian car number plate format** using a custom dataset.

---

## Dataset

The dataset used for training is from **Roboflow**:

- **Dataset Name**: [License Plate Detector - Indian Format](https://universe.roboflow.com/mochoye/license-plate-detector-ogxxg)
- **Format**: Annotations include bounding boxes for the license plates.
- **Content**: Images of vehicles with Indian number plates.

You can download the dataset and augment it for your needs, ensuring that it contains a sufficient number of images in various lighting conditions, angles, and occlusions.

---

## Installation

To get started with the project, follow these steps:

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/indian-anpr.git
cd indian-anpr
