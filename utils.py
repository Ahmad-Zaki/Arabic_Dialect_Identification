import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from typing import Any

def confusion_matrix(y_train_true: Any, y_train_pred: Any, y_val_true: Any, y_val_pred: Any) -> None:
    """Display the cofusion matrix of train and validation splits.
    
    Parameters
    ----------
    y_train_true:
        True labels of training data.
    
    y_train_pred:
        Predicted labels of training data.

    y_val_true:
        True labels of validation data.
    
    y_val_pred:
        Predicted labels of validation data.
    """

    fig, ax = plt.subplots(1, 2, figsize = (15, 5))

    ax[0].set_title("Training Set Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_train_true, y_train_pred,
                                            ax = ax[0], 
                                            cmap = "cividis", 
                                            xticks_rotation = "vertical")
    
    ax[1].set_title("Validation Set Confusion Matrix")
    ConfusionMatrixDisplay.from_predictions(y_val_true, y_val_pred, 
                                            ax = ax[1],
                                            cmap = "cividis", 
                                            xticks_rotation = "vertical")

    plt.show()


def report(split: str, y_true: Any, y_pred: Any, precision: int = 4) -> None:
    """Print the classification report.
    
    Parameters
    ----------
    split: str
        Name of the set this predictions belong to (training, validation, testing)

    y_true:
        True labels of the data.
    
    y_pred:
        Predicted labels of the data.

    precision: int
        Number of digits for formatting output floating point values.
        Default = 4 
    """

    print(f"  {split} set Classification Report:")
    print(classification_report(y_true, y_pred, digits = precision))