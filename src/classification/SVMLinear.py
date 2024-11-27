from sklearn.svm import LinearSVC
from typing import Literal

from .ClassificationModel import ClassificationModel


class SVMLinearModel(ClassificationModel):
    def __init__(self, C=1.0, class_weight: None | dict | Literal['balanced']='balanced', model_dir='./models/'):
        """
        Initialize the SVM model with given hyperparameters.

        Args:
            kernel (str): Kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid').
            C (float): Penalty parameter of the error term.
            model_dir (str): The path to the directory where the model be saved/loaded from.
        """
        model = LinearSVC(C=C, class_weight=class_weight)  # Class weights are automatically assigned to account for unbalance

        super().__init__(model, model_dir=model_dir, model_name='SVMLinear')