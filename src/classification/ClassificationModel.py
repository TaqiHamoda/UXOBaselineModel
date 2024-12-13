import numpy as np
import os, pickle, datetime

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem


class _ClassificationModel:
    def __init__(self, model):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


class ClassificationModel:
    def __init__(self, model, model_dir: str = './models/', model_name: str = '', standardize: bool = True, pca: bool = True, kernel_mapping: bool = True, n_components: int = 100):
        self.model_dir = model_dir
        self.model_name = model_name

        if not os.path.exists(self.model_dir) or not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        model_name_dir = os.path.join(self.model_dir, self.model_name)
        if not os.path.exists(model_name_dir) or not os.path.isdir(model_name_dir):
            os.mkdir(model_name_dir)

        self.model_dir = model_name_dir

        pipeline = []
        if standardize:
            pipeline.append(('scaling', StandardScaler()))

        if kernel_mapping:
            pipeline.append(('kernel', Nystroem(n_jobs=-1, n_components=n_components)))

        if pca:
            pipeline.append(('pca', PCA()))

        pipeline.append(('training', _ClassificationModel(model)))

        self.model = Pipeline(pipeline, verbose=True)


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the SVM model on the provided data.

        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
        """
        return self.model.fit_transform(X_train, y_train,)


    def evaluate(self, X: np.ndarray):
        """
        Evaluate the SVM model on the given data.

        Args:
            X (np.array): Input features.

        Returns:
            np.array: Prediction for y vector
        """
        return self.model.transform(X)


    def save_model(self) -> None:
        """
        Save the trained model to a file.

        Args:
            filename: Path to save the model file.
        """
        timestamp = datetime.datetime.now().isoformat()
        filename = f"{self.model_name}-{timestamp}.pkl"

        model_path = os.path.join(self.model_dir, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)


    def load_model(self, filename: str) -> None:
        """
        Load a model from a file.

        Args:
            filename: Path to load the model file.
        """
        model_path = os.path.join(self.model_dir, filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise FileNotFoundError(f"No model file found at {model_path}")
