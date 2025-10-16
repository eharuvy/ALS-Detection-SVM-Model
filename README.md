# ALS Detection SVM Model
*(Originally developed in R and subsequently translated to MATLAB)*

## Overview
This MATLAB project implements a two-fold (half-half) cross-validation Support Vector Machine (SVM) model to classify ALS patients versus healthy controls using 22 provided CSV datasets.

## Pipeline
1. Loads CSV files for ALS and healthy subjects into MATLAB tables.  
2. Adds a label column to indicate class membership: ALS (1) or Healthy (0).  
3. Combines all subjects into a single data table for analysis.  
4. Performs two-fold (half-half) cross-validation by randomly splitting each class into two halves.  
5. Trains an SVM with a Radial Basis Function (RBF) kernel for each fold, applying class weights to balance ALS and Healthy samples.  
6. Computes labeled confusion matrices for each cross-validation iteration and a summed confusion matrix across both iterations.  
7. Calculates the average accuracy, sensitivity, and specificity from both folds.

## Folder Structure
ALSModelSubmission/  
│  
├── ALSDetection_model.m — Main MATLAB script  
├── data/ — Folder containing CSV datasets  
│   ├── A01.csv ... A11.csv — ALS subject data files  
│   └── N01.csv ... N11.csv — Healthy subject data files  

---

## Instructions to Run

1.  Open **MATLAB**.
2.  Navigate to the folder that contains `ALSDetection_model.m` and the `data/` folder.
3.  Run the script: `ALSDetection_model`

### Expected Output

The script will output:
* Confusion matrices for each cross-validation iteration (as labeled MATLAB tables).
* Summed confusion matrix across both iterations.
* Average performance metrics: **Accuracy, Sensitivity, Specificity**.

---

## Notes on the Implementation

### SVM Parameters
* **Kernel:** Radial Basis Function (RBF)
* **BoxConstraint (cost):** $80$
* **Features:** Standardized
* **Class Weighting:** Applied to balance ALS vs Healthy samples.

### Cross-validation
* **Type:** Two-fold (half-half) cross-validation.
* **Folding:** Each class (ALS, Healthy) is independently split into two equal halves. Folds are swapped in iteration 2 (e.g., Train on Fold 1/Test on Fold 2, then Train on Fold 2/Test on Fold 1).

---

## Author

**Ethan Haruvy**
UGS 303 – ALS Machine Learning Project
