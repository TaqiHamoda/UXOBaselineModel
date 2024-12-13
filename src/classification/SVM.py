from sklearn.svm import LinearSVC
from typing import Literal

from .ClassificationModel import ClassificationModel


class SVMModel(ClassificationModel):
    def __init__(self, C: float = 1.0, class_weight: None | dict | Literal['balanced'] = 'balanced', model_dir: str = './models/', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        model = LinearSVC(C=C, class_weight=class_weight)  # Class weights are automatically assigned to account for unbalance

        super().__init__(model, model_dir=model_dir, model_name='SVM', standardize=standardize, pca=pca, kernel_mapping=kernel_mapping, n_components=n_components)