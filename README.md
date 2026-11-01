# Decision Tree - Wine Classification

## P1 - Entropy
Implemented an entropy function using NumPy and tested it with `[50, 30]`.

Expected entropy: approximately `0.9544`.

## P2 - Max Depth
Trained Decision Tree models on the Iris dataset using `max_depth` values from 1 to 8 and printed training accuracy for each depth.

## Mini Project - Wine Dataset
- Loaded the Wine dataset from Scikit-learn.
- Split the data into training and validation sets.
- Trained Decision Tree classifiers with max_depth from 1 to 8.
- Calculated training and validation accuracy.
- Plotted both accuracies against max_depth.
- Selected the best depth based on validation accuracy.

## Requirements
- NumPy
- Matplotlib
- Scikit-learn

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python decision_tree_wine.py
```

The program displays the accuracy results and saves the plot as `wine_accuracy.png`.
