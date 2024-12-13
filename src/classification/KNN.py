from typing import Literal
from sklearn.neighbors import KNeighborsClassifier

from .ClassificationModel import ClassificationModel


class KNNModel(ClassificationModel):
    def __init__(self, n_neighbors: int = 5, weights: Literal['unifrom', 'distance'] = 'distance', model_dir: str = './models/', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        super().__init__(model, model_dir=model_dir, model_name='KNN', standardize=standardize, pca=pca, kernel_mapping=kernel_mapping, n_components=n_components)
