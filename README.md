<div align="center">

# Applied Machine Learning

*Exploring classification, ensemble learning, model evaluation, and neural networks*

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classification-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Optional%20deep%20learning-FF6F00.svg)

[Overview](#overview) • [Projects](#projects) • [Methods](#methods) • [Setup](#setup) • [Results](#results) • [Repository Structure](#repository-structure)

</div>

---

## Overview

This repository brings together a series of practical machine learning projects developed as part of a master's-level course. It covers the full workflow from preparing datasets and training classifiers to comparing models and evaluating their performance.

The work combines **classical machine learning** with a small **deep learning** experiment. It is intended as a showcase of techniques explored through coursework, rather than as a packaged application or production-ready library.

## Projects

| Folder | Focus | Data |
|---|---|---|
| [`A1_2/`](./A1_2/) | Decision trees, bootstrap sampling, bagging, and ensemble experiments | Wine quality |
| [`A2_2/`](./A2_2/) | Support Vector Machines, cross-validation, kernel comparison, and a perceptron implementation | Diabetes classification |
| [`A3_2/`](./A3_2/) | Decision-tree evaluation across sampled training sets and repeated test samples | College outcomes |
| [`deeplearning/`](./deeplearning/) | Fully connected neural networks for handwritten Hiragana classification | Hiragana image dataset |
| [`final/`](./final/) | Feature scaling and SVM hyperparameter search for a final classification task | Compressed feature and label files |

## Methods

### Decision trees and ensembles

- Train and evaluate decision-tree classifiers.
- Explore the effect of training sample size and tree hyperparameters.
- Build bootstrap-based bagging experiments and compare them with random forests.
- Measure variation in test accuracy across repeated samples and estimate confidence intervals.

### Support Vector Machines

- Compare linear, polynomial, RBF, and sigmoid kernels.
- Estimate performance with 10-fold cross-validation.
- Use grid search for hyperparameter selection in the final classification experiment.
- Implement a basic perceptron training routine.

### Neural networks

The deep learning notebook uses TensorFlow/Keras to classify 28 × 28 Hiragana images. It prepares labels for categorical classification and compares fully connected networks with different hidden-layer activations.

## Setup

Use Python 3 and install the libraries needed for the classical machine learning scripts:

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

For the notebook, install Jupyter and TensorFlow as well:

```bash
pip install jupyter tensorflow
```

Run each script from its corresponding folder because the scripts load data using paths relative to the current working directory.

## Usage

### Wine quality experiments

```bash
cd A1_2
python q_1.py
python q_2.py
python q_3.py
python q_4.py
```

### SVM and perceptron experiments

```bash
cd A2_2
python q1.py
```

### College outcome evaluation

```bash
cd A3_2
python 1.py
python 2.py
python 3.py
```

### Hiragana neural network notebook

Start Jupyter from the `deeplearning/` folder so the notebook can find its data files:

```bash
cd deeplearning
jupyter notebook main.ipynb
```

### Final classification experiment

```bash
cd final
python final.py
```

The scripts print metrics and experiment summaries to the terminal. The final project report is available at [`final/report.pdf`](./final/report.pdf).

## Results

The repository contains experiment code and a final report. Some results are produced when the scripts or notebook are run; no single benchmark score is presented here because scores depend on the experiment and its evaluation setup.

## Repository Structure

```text
ml_master_course_q2/
├── A1_2/
│   ├── wine_train.csv
│   ├── wine_test.csv
│   └── *.py                  # Decision trees and ensemble experiments
├── A2_2/
│   ├── diabetes_train.csv
│   ├── diabetes_test.csv
│   └── q1.py                 # SVM and perceptron experiments
├── A3_2/
│   ├── college_train.csv
│   ├── college_test.csv
│   └── 1.py, 2.py, 3.py      # Repeated-sample evaluation
├── deeplearning/
│   ├── A4_data-20240409/     # Hiragana image and label arrays
│   └── main.ipynb            # TensorFlow/Keras experiments
├── final/
│   ├── A5_2024_*.gz          # Compressed feature and label data
│   ├── final.py
│   └── report.pdf
└── README.md
```

## Notes

- These are course experiments, and scripts are organized by assignment rather than behind a shared command-line interface.
- Run each script from the folder containing its input data; otherwise, its relative data paths may not resolve.
- No license file is included. Check the terms that apply to the coursework and datasets before reusing or redistributing them.
