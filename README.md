# Email Spam Detection using Machine Learning

This repository contains a machine learning project dedicated to classifying emails as *spam* or *ham* (non-spam) using the [Spambase dataset](https://archive.ics.uci.edu/ml/datasets/Spambase) from the UCI Machine Learning Repository.

## Overview

The goal of this project is to implement and evaluate several classic machine learning algorithms to detect spam. Given statistical patterns in the text—such as word and character frequencies, and capital letter runs—the classifiers are trained to distinguish between legitimate and unsolicited email.

### Project Details

*   **Course Assignment:** Machine Learning CA-2 Project
*   **Institution:** KIET Deemed to Be University
*   **Contributors:**
    *   Misty Jangid
    *   Milind Pushp

## Features & Methodology

The Spambase dataset consists of 4,601 email records with 57 numerical features. Our implementation workflow follows:
1.  **Exploratory Data Analysis (EDA):** Checking class distribution and identifying key features.
2.  **Preprocessing:** Removing duplicates, feature scaling via `StandardScaler`, and executing an 80/20 stratified train-test split.
3.  **Model Training:** Training six distinct classifiers:
    *   Logistic Regression
    *   Support Vector Machine (Linear & RBF Kernels)
    *   Random Forest (Best Performer ✨)
    *   K-Nearest Neighbours (KNN)
    *   Naïve Bayes
4.  **Evaluation:** Comparing architectures with metrics including Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

## Results Highlights

*   The **Random Forest** algorithm performed the best, achieving an accuracy of **~94.30%** and an ROC-AUC score of **~0.99**. 
*   Top indicators for spam included features relating to total capital run length, the word "free", and the exclamation mark character (`!`).

## Repository Structure

*   `Spam_detection.ipynb` / `Spam_Detection_with_Visuals.ipynb` — Main Jupyter Notebooks containing the full code pipeline.
*   `CA2_Report.pdf` (and `CA2_Report.tex` / `.md`) — The detailed structured report documenting findings, methodologies, and analysis.
*   `spambase.data` / `spambase.names` / `spambase.DOCUMENTATION` — Sourced datasets and metadata mapping files.

## Getting Started

1.  Clone this repository to your local machine.
2.  Ensure you have Python and standard libraries installed.
3.  Install necessary data science requirements (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
4.  Launch the Jupyter Notebook environment:
    ```bash
    jupyter notebook
    ```
5.  Open the notebook file (`Spam_Detection_with_Visuals.ipynb` or `Spam_detection.ipynb`) and execute the cells to reproduce the analysis.
