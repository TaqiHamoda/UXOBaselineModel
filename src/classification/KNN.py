from typing import Literal
from sklearn.neighbors import KNeighborsClassifier

from .ClassificationModel import ClassificationModel


class KNNModel(ClassificationModel):
    def __init__(self, n_neighbors: int = 5, weights: Literal['unifrom', 'distance'] = 'distance', model_dir: str = './models/'):
        """
        Initialize the KNN model with given hyperparameters.

        Args:
            n_neighbors (int): Number of neighbors to use.
            weights (str): Weight function used in prediction ('uniform' or 'distance').
            model_dir (str): The path to the directory where the model be saved/loaded from.
        """

        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

        super().__init__(model, model_dir=model_dir, model_name='KNN')
