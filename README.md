## repository structure

configs/      configuration files for experiments

src/          model training, evaluation, and utilities

notebooks/    exploratory development notebooks

experiments/  experiment summary and comparison table

outputs/      generated checkpoints, logs, and metrics

figures/      plots used in the report

reports/      final report and discussion

tests/        reproducibility and validation tests

slurm/        Snellius job scripts


# PneumoniaMNIST medical ai project

## selected task
We use PneumoniaMNIST from MedMNIST, a binary medical image classification task for detecting pneumonia from chest x-ray images.

## data splits
We use the official MedMNIST train, validation, and test splits. The validation set is used for model selection and hyperparameter tuning. The test set is used only once for final evaluation.

## preprocessing
Images are converted to tensors and normalised. For logistic regression, images are flattened into feature vectors. For the CNN, images remain as 1 x 28 x 28 tensors.

## models
Baseline: logistic regression using scikit-learn.  
Deep learning model: small convolutional neural network using PyTorch.

## training and validation
The CNN is trained with CrossEntropyLoss and Adam. Validation performance is checked after each epoch. The model is evaluated on it, and the checkpoint with the highest validation AUC is saved as best_model.pt and used for final test evaluation.

## evaluation
Both models are evaluated using the same test split and reported with accuracy, F1-score, AUC, and confusion matrix.

## main results
model                    | accuracy        | f1               | auc               | sensitivity      | specificity
logistic_regression,     0.8733974358974359,0.9056152927120669,0.9297720797720798,0.9717948717948718,0.7094017094017094
small_cnn,               0.8717948717948718,0.9061032863849765,0.9336675432829279,0.9897435897435898,0.6752136752136753
small_cnn_class_weighted,0.8814102564102564,0.9121140142517815,0.9359248301555994,0.9846153846153847,0.7094017094017094

NB: Raw metrics are stored in outputs/*.json. A summary comparison is provided in experiments/final_model_comparsion.csv.


## interpretation
The weighted CNN achieved the highest AUC (0.936) and accuracy (0.881). The standard CNN slightly improved AUC over logistic regression but not accuracy. The results suggest that class weighting helps the CNN focus on minority-class examples. Performance is likely limited by the small 28x28 image resolution and the relatively simple architecture.

## reproducibility

...
install requirements:
pip install -r requirements.txt

run baseline:
python src/train_baseline.py --config configs/baseline.yaml

run cnn:
python src/train_cnn.py --config configs/cnn.yaml

run weighted cnn:
python src/train_cnn_weighted.py --config configs/cnn_weighted.yaml

evaluate:
python src/evaluate.py --checkpoint outputs/cnn_run/best_model.pt --split test

run on snellius:
sbatch slurm/train_cnn.slurm

## hardware and execution

The code automatically uses a GPU when one is available. Training can be executed locally, in Google Colab, or on Snellius. Example SLURM job scripts for Snellius are provided in the `slurm/` directory.
