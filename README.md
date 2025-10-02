# DigitalMarket-Demo
Project Overview

This repository presents a demo of an image classification system using a Vision Transformer (ViT) to classify items in a market ("bozor" in Uzbek) context, such as brooms and bread. Developed as a portfolio piece to showcase enterprise-level computer vision expertise, it is a simplified version of larger-scale projects. The system processes images to identify and classify market items, leveraging a pre-trained ViT model fine-tuned on a custom dataset. Large datasets (e.g., thousands of labeled images) are not committed due to size constraints, but a small demo dataset (e.g., broom and bread photos) is referenced to illustrate functionality.

The project demonstrates proficiency in deep learning, dataset management, and model training/evaluation, tailored for applications like inventory management or automated market analysis.

Technologies Used





Python 3.8+: Core language for its robust ecosystem and flexibility in machine learning workflows.



Hugging Face Transformers: Provides pre-trained ViT models and utilities for image classification.



PyTorch: Backend for model training and inference, supporting GPU acceleration.



Hugging Face Datasets: Streamlines dataset loading and preprocessing for image classification tasks.



PIL (Pillow): Handles image loading and format conversion.



Roboflow: Manages dataset acquisition and versioning.



OS, Shutil, Random: Facilitates file operations and dataset splitting.



Pathlib: Ensures cross-platform file path compatibility.

Why These Technologies?





Vision Transformer (ViT): Chosen for its superior performance in image classification, leveraging attention mechanisms to capture global context in images, ideal for distinguishing market items with varied appearances.



Hugging Face Transformers: Offers pre-trained ViT models (google/vit-base-patch16-224-in21k) and a user-friendly API, reducing development time and enabling fine-tuning for custom classes.



PyTorch: Selected for its dynamic computation graph and GPU support, critical for efficient training and inference.



Hugging Face Datasets: Simplifies loading and managing image datasets in a structured format, supporting train/validation splits.



Roboflow: Enables easy dataset download and versioning, mimicking enterprise workflows where datasets are centrally managed.



PIL: Used for reliable image processing, ensuring compatibility with ViT's input requirements.



File Utilities (OS, Shutil, Random): Essential for organizing datasets into train/validation splits, a common preprocessing step in enterprise pipelines.

Implementation Details

The system is modular, with scripts addressing distinct tasks:





Dataset Download (download-dataset.py):





Purpose: Fetches labeled image datasets from Roboflow (e.g., "bozor-classification-2" project, version 8).



How: Uses Roboflow's API to download datasets in a folder structure compatible with image classification tasks.



Why: Ensures access to structured, annotated data, reflecting enterprise practices for dataset management. The demo references a small dataset (e.g., broom and bread photos), while the full version included thousands of images.



Dataset Splitting (splittingData.py):





Purpose: Organizes images into train (80%) and validation (20%) sets.



How: Randomly shuffles images per class and copies them to dataset_split/train and dataset_split/val folders, preserving class subfolders.



Why: Enables robust model training and evaluation, ensuring the model generalizes well to unseen data.



Model Training (vitrain.py):





Purpose: Fine-tunes a pre-trained ViT model on the custom dataset for market item classification.



How: Loads the dataset using Hugging Face Datasets, preprocesses images with ViTImageProcessor, and trains with the Trainer API. Configures hyperparameters (e.g., epochs=10, learning_rate=5e-5) and saves the model to ./vit_model.



Why: Adapts ViT to specific classes (e.g., broom, bread), achieving high accuracy for market-specific tasks.



Inference (inference.py):





Purpose: Classifies images using the trained ViT model.



How: Processes single images or entire folders, using ViTImageProcessor for input preparation and outputting predicted class labels (e.g., "broom" or "bread").



Why: Demonstrates the model's practical application for identifying market items, a key feature in inventory or quality control systems.

Design Principles





Modularity: Scripts are independent, mirroring enterprise microservices for data prep, training, and inference.



Scalability: Designed to handle large datasets (thousands of images in the full version), though the demo uses a small subset (e.g., broom and bread).



Extensibility: Easily adaptable to new classes by updating the dataset and retraining.



Robustness: Includes error handling for file operations and model loading, ensuring reliability.

Limitations





Dataset: The full dataset (thousands of images) is not committed due to size limits. The demo uses example classes (e.g., broom, bread), downloadable via download-dataset.py.



Paths: Hardcoded for demo purposes (e.g., C:\Users\elbek\...); production versions use config files or environment variables.



Scope: Focuses on offline image classification; enterprise versions included real-time processing and integration with market systems.
