import numpy as np
import matplotlib.pyplot as plt
import json
from tabulate import tabulate
from tqdm import tqdm

plt.style.use("dark_background")


class RobustScaler:
    def __init__(self, normalize=True):
        self.normalize = normalize
        self.median_ = {}
        self.iqr_ = {}

    def fit(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError("Input should be a numpy array.")
        X = np.array(X, dtype=float)
        for i in range(X.shape[0]):
            row = X[i, :]
            self.median_[i] = np.nanmedian(row)
            Q1 = np.nanpercentile(row, 25)
            Q3 = np.nanpercentile(row, 75)
            self.iqr_[i] = Q3 - Q1

    def transform(self, X):
        if self.normalize:
            if not self.median_ or not self.iqr_:
                raise RuntimeError("You must fit the scaler before transforming data.")
            if not isinstance(X, np.ndarray):
                raise TypeError("Input should be a numpy array.")
            X = np.array(X, dtype=float)
            X_scaled = np.zeros_like(X)
            for i in range(X.shape[0]):
                X_scaled[i, :] = (X[i, :] - self.median_[i]) / self.iqr_[i]
            return X_scaled
        else:
            return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_scaled):
        if self.normalize:
            if not self.median_ or not self.iqr_:
                raise RuntimeError(
                    "You must fit the scaler before inverse transforming data."
                )
            if not isinstance(X_scaled, np.ndarray):
                raise TypeError("Input should be a numpy array.")
            X_scaled = np.array(X_scaled, dtype=float)
            X_original = np.zeros_like(X_scaled)
            for i in range(X_scaled.shape[0]):
                X_original[i, :] = (X_scaled[i, :] * self.iqr_[i]) + self.median_[i]
            return X_original
        else:
            return X_scaled


class DeepNeuralNetwork:
    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_layers=(16, 16, 16),
        metadata=None,
        normalize=True,
        seed=1,
    ):
        if X_train.shape[0] != X_test.shape[0] or y_train.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"Number of variables and number of labels must match across training and testing sets. "
                f"Shapes: X_train{X_train.shape}, X_test{X_test.shape}, y_train{y_train.shape}, y_test{y_test.shape}"
            )

        if y_train.shape[0] != 1:
            raise ValueError(
                f"Number of labels is {y_train.shape[0]} while 1 is expected"
            )

        np.random.seed(seed)

        self.dimensions = [X_train.shape[0]] + list(hidden_layers) + [y_train.shape[0]]
        self.robust_scaler = RobustScaler(normalize=normalize)

        if metadata:
            self.import_metadata(metadata)
        else:
            self.robust_scaler.fit(X_train)
            self.parameters = self.init(self.dimensions)

        self.X_train, self.y_train = self.robust_scaler.transform(X_train), y_train
        self.X_test, self.y_test = self.robust_scaler.transform(X_test), y_test

        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

    def train(
        self,
        learning_rate=1e-2,
        l1_lambda=0.0,  # L1 regularization strength
        l2_lambda=0.0,  # L2 regularization strength
        dropout_rate=0.0,
        n_iter=100000,
        metrics=1000,
        early_stopping=False,  # Enable/Disable automatic overfitting detection
        overfitting_tolerance=5,  # Number of iterations to wait before stopping
        slope_window=10,  # Window size to calculate the test loss slope
    ):
        metrics_spacing = n_iter / metrics
        tolerance_counter = 0

        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        def metrics(parameters, activations):
            self.train_loss.append(
                self.log_loss(
                    self.y_train,
                    activations,
                    parameters=parameters,
                    l1_lambda=l1_lambda,
                    l2_lambda=l2_lambda,
                )
            )
            self.train_accuracy.append(self.accuracy_score(self.y_train, activations))

            activations_test = self._forward_propagation(self.X_test, parameters)
            self.test_loss.append(
                self.log_loss(
                    self.y_test,
                    activations_test,
                    parameters=parameters,
                    l1_lambda=l1_lambda,
                    l2_lambda=l2_lambda,
                )
            )
            self.test_accuracy.append(
                self.accuracy_score(self.y_test, activations_test)
            )

        for i in tqdm(range(n_iter)):
            activations = self._forward_propagation(
                self.X_train, self.parameters, dropout_rate=dropout_rate
            )
            gradients = self._backward_propagation(
                self.y_train,
                self.parameters,
                activations,
                l1_lambda,
                l2_lambda,
            )
            self.parameters = self._update(gradients, self.parameters, learning_rate)

            if i % metrics_spacing == 0:
                metrics(self.parameters, activations)

            if early_stopping:
                if i % overfitting_tolerance == 0:
                    if len(self.test_loss) >= slope_window:
                        recent = np.array(self.test_loss[-slope_window:])
                        x = np.arange(len(recent))
                        slope = np.polyfit(x, recent, 1)[0]

                        if slope > 0:
                            tolerance_counter += 1
                        else:
                            tolerance_counter = 0
                        if tolerance_counter >= 1:
                            print(
                                f"Overfitting detected. Training stopped at iteration {i}."
                            )
                            break

    def export_metadata(self, indent=None) -> str:
        metadata = {
            "dimensions": self.dimensions,
            "parameters": {k: v.tolist() for k, v in self.parameters.items()},
            "normalize": self.robust_scaler.normalize,
            "robust_scaler": {
                "median": {
                    str(k): v.tolist() for k, v in self.robust_scaler.median_.items()
                },
                "IQR": {str(k): v.tolist() for k, v in self.robust_scaler.iqr_.items()},
            },
        }
        return json.dumps(metadata, indent=indent)

    def import_metadata(self, metadata_json: str):
        metadata = json.loads(metadata_json)

        self.parameters = {k: np.array(v) for k, v in metadata["parameters"].items()}

        self.robust_scaler.normalize = metadata["normalize"]
        self.robust_scaler.median_ = {
            int(k): np.float64(v)
            for k, v in metadata["robust_scaler"]["median"].items()
        }
        self.robust_scaler.iqr_ = {
            int(k): np.float64(v) for k, v in metadata["robust_scaler"]["IQR"].items()
        }

    @property
    def metrics(self):
        activations_train = self._forward_propagation(self.X_train, self.parameters)
        activations_test = self._forward_propagation(self.X_test, self.parameters)

        y_train_pred = activations_train["A" + str(len(activations_train) - 1)]
        y_test_pred = activations_test["A" + str(len(activations_test) - 1)]

        train_loss = self.log_loss(self.y_train, activations_train)
        test_loss = self.log_loss(self.y_test, activations_test)

        train_loss_slope = (
            self.slope_log(self.train_loss, window=10) if self.train_loss else None
        )
        test_loss_slope = (
            self.slope_log(self.test_loss, window=10) if self.test_loss else None
        )

        train_accuracy = self.accuracy_score(self.y_train, activations_train)
        test_accuracy = self.accuracy_score(self.y_test, activations_test)

        train_f1_score = self.f_score(1, self.y_train, y_train_pred)
        test_f1_score = self.f_score(1, self.y_test, y_test_pred)

        train_auc = self.auc_roc(self.y_train, y_train_pred)
        test_auc = self.auc_roc(self.y_test, y_test_pred)

        train_mcc_score = self.mcc_score(self.y_train, y_train_pred)
        test_mcc_score = self.mcc_score(self.y_test, y_test_pred)

        if train_accuracy <= 0.75 and test_accuracy <= 0.75:
            fit_score = -(train_accuracy + test_accuracy - 1) / 2
        else:
            fit_score = max(0, train_accuracy - test_accuracy)

        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_loss_slope": train_loss_slope,
            "test_loss_slope": test_loss_slope,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_f1_score": train_f1_score,
            "test_f1_score": test_f1_score,
            "train_auc": train_auc,
            "test_auc": test_auc,
            "train_mcc_score": train_mcc_score,
            "test_mcc_score": test_mcc_score,
            "fit_score": fit_score,
        }

    @staticmethod
    def init(dimensions):
        parameters = {}
        C = len(dimensions)

        for c in range(1, C):
            parameters["W" + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
            parameters["b" + str(c)] = np.random.randn(dimensions[c], 1)

        return parameters

    @staticmethod
    def _forward_propagation(X, parameters, dropout_rate=0.0):
        activations = {"A0": X}

        C = len(parameters) // 2

        for c in range(1, C + 1):

            Z = (
                parameters["W" + str(c)].dot(activations["A" + str(c - 1)])
                + parameters["b" + str(c)]
            )
            A = 1 / (1 + np.exp(-Z))

            if c < C and dropout_rate > 0.0:
                D = np.random.rand(A.shape[0], A.shape[1]) > dropout_rate
                A = A * D
                A = A / (1 - dropout_rate)

            activations["A" + str(c)] = A
        return activations

    @staticmethod
    def _backward_propagation(y, parameters, activations, l1_lambda, l2_lambda):
        m = y.shape[1]
        C = len(parameters) // 2

        dZ = activations["A" + str(C)] - y
        gradients = {}

        for c in reversed(range(1, C + 1)):
            gradients["dW" + str(c)] = (
                1 / m * np.dot(dZ, activations["A" + str(c - 1)].T)
            )
            gradients["db" + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if l2_lambda > 0:
                gradients["dW" + str(c)] += (l2_lambda / m) * parameters["W" + str(c)]
            if l1_lambda > 0:
                gradients["dW" + str(c)] += (l1_lambda / m) * np.sign(
                    parameters["W" + str(c)]
                )
            if c > 1:
                dZ = (
                    np.dot(parameters["W" + str(c)].T, dZ)
                    * activations["A" + str(c - 1)]
                    * (1 - activations["A" + str(c - 1)])
                )

        return gradients

    @staticmethod
    def _update(gradients, parameters, learning_rate):

        C = len(parameters) // 2

        for c in range(1, C + 1):
            parameters["W" + str(c)] = (
                parameters["W" + str(c)] - learning_rate * gradients["dW" + str(c)]
            )
            parameters["b" + str(c)] = (
                parameters["b" + str(c)] - learning_rate * gradients["db" + str(c)]
            )

        return parameters

    def predict(self, X, scaling=True):
        activations = DeepNeuralNetwork._forward_propagation(
            self.robust_scaler.transform(X) if scaling else X, self.parameters
        )
        return activations["A" + str(len(activations) - 1)]

    @staticmethod
    def log_loss(
        y, activations, parameters=None, l1_lambda=0.0, l2_lambda=0.0, epsilon=1e-15
    ):
        Af = activations["A" + str(len(activations) - 1)]
        Af = np.clip(Af, epsilon, 1 - epsilon)
        m = y.size

        y = y.flatten()
        Af = Af.flatten()
        log_loss_value = (1 / m) * np.sum(-y * np.log(Af) - (1 - y) * np.log(1 - Af))

        if parameters is not None:
            l1_regularization = 0
            l2_regularization = 0
            C = len(parameters) // 2

            for c in range(1, C + 1):
                W = parameters["W" + str(c)]
                # L1 regularization term: sum of absolute values of weights
                l1_regularization += np.sum(np.abs(W))
                # L2 regularization term: sum of squared values of weights
                l2_regularization += np.sum(np.square(W))

            log_loss_value += (l1_lambda / m) * l1_regularization
            log_loss_value += (l2_lambda / (2 * m)) * l2_regularization
        return log_loss_value

    @staticmethod
    def accuracy_score(y, activations):
        Af = activations["A" + str(len(activations) - 1)]
        y = y.flatten()
        Af = Af.flatten()
        return np.mean((Af >= 0.5) == y)

    # metrics
    @staticmethod
    def binary_classification(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # True Positives
        tp = np.sum((y_pred >= 0.5) & (y_true == 1))
        # False Positives
        fp = np.sum((y_pred >= 0.5) & (y_true == 0))
        # True Negatives
        tn = np.sum((y_pred < 0.5) & (y_true == 0))
        # False Negatives
        fn = np.sum((y_pred < 0.5) & (y_true == 1))

        return tp, fp, tn, fn

    @staticmethod
    def auc_roc(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = np.array(y_true)[sorted_indices]
        y_pred_sorted = np.array(y_pred)[sorted_indices]

        tpr_values = []
        fpr_values = []

        P = sum(y_true)
        N = len(y_true) - P

        if P == 0 or N == 0:
            raise ValueError("y_true contains no positive or negative samples.")

        TP = 0
        FP = 0

        for i in range(len(y_pred_sorted)):
            if y_true_sorted[i] == 1:
                TP += 1
            else:
                FP += 1

            TPR = TP / P if P > 0 else 0
            FPR = FP / N if N > 0 else 0

            tpr_values.append(TPR)
            fpr_values.append(FPR)

        auc = 0.0
        for i in range(1, len(fpr_values)):
            auc += (
                (fpr_values[i] - fpr_values[i - 1])
                * (tpr_values[i] + tpr_values[i - 1])
                / 2
            )

        return auc

    @staticmethod
    def f_score(beta, y_true, y_pred):
        tp, fp, tn, fn = DeepNeuralNetwork.binary_classification(y_true, y_pred)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return (1 + beta) * ((precision * recall) / (beta**2 * precision + recall))

    @staticmethod
    def mcc_score(y_true, y_pred):
        tp, fp, tn, fn = DeepNeuralNetwork.binary_classification(y_true, y_pred)
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        return mcc

    @staticmethod
    def slope_log(history, window=10):
        if len(history) < window:
            raise ValueError(
                f"{window} values are provided while {len(history)} are required"
            )
        recent = np.array(history[-window:])
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        if 0 < slope <= 1:
            return -np.log(slope)
        elif -1 <= slope < 0:
            return np.log(-slope)
        else:
            raise ValueError(
                "The slope is too steep to calculate the signed logarithm. It must be between -1 and 1 (without 0)"
            )

    @staticmethod
    def slope(history, window=10):
        if len(history) < window:
            raise ValueError(
                f"{window} values are provided while {len(history)} are required"
            )
        recent = np.array(history[-window:])
        x = np.arange(len(recent))
        return np.polyfit(x, recent, 1)[0]

    def X_original(self, X):
        return self.robust_scaler.inverse_transform(X)

    def show_progress(self):

        plt.plot(self.train_loss, "b-", label="Train Loss")
        plt.plot(self.test_loss, "r-", label="Test Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss")
        plt.legend()
        plt.show()

        plt.plot(self.train_accuracy, "b-", label="Train Accuracy")
        plt.plot(self.test_accuracy, "r-", label="Test Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuraccy")
        plt.legend()
        plt.show()

    def show_stats(self):
        for k, v in self.metrics.items():
            print(k, ": ", v)
        print()
        print("RobustScaler parameters:")
        data = []
        for i in range(0, len(self.robust_scaler.median_)):
            data.append(
                [f"x{i + 1}", self.robust_scaler.median_[i], self.robust_scaler.iqr_[i]]
            )

        headers = ["input", "median", "IQR"]
        table = tabulate(data, headers=headers)
        print(table)
        print()

        data = []
        for c in range(1, len(self.parameters) // 2 + 1):
            W = self.parameters["W" + str(c)]
            b = self.parameters["b" + str(c)]
            for i in range(W.shape[0]):
                row = [i + 1, c, b[i, 0], W.shape[1]]
                row.extend(W[i, :])
                data.append(row)

        headers = ["Neuron", "Layer", "b", "Inputs"] + [
            f"w{j + 1}"
            for j in range(
                max(
                    W.shape[1]
                    for W in self.parameters.values()
                    if isinstance(W, np.ndarray)
                )
            )
        ]
        table = tabulate(data, headers=headers, floatfmt=".4f")
        print(table)

    def show2d_predictions(
        self,
        feature_x=0,
        feature_y=1,
        cmap=plt.cm.Spectral,
        levels=None,
        colorbar=False,
        proba=False,
        transparency=False,
    ):
        if self.X_train.shape[0] < 2:
            raise ValueError("The dataset must have at least 2 features.")

        if feature_x < 0 or feature_x >= self.X_train.shape[0]:
            raise ValueError(
                f"feature_x index {feature_x} is out of bounds. It must be between 0 and {self.X_train.shape[0] - 1}."
            )

        if feature_y < 0 or feature_y >= self.X_train.shape[0]:
            raise ValueError(
                f"feature_y index {feature_y} is out of bounds. It must be between 0 and {self.X_train.shape[0] - 1}."
            )

        X = self.X_original(self.X_train)

        # Get min and max for the selected dimensions
        x_min, x_max = X[feature_x, :].min() - 1, X[feature_x, :].max() + 1
        y_min, y_max = X[feature_y, :].min() - 1, X[feature_y, :].max() + 1

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01)
        )

        # Create a new array to hold the grid points
        num_samples = xx.size  # Total number of points in the grid
        grid = np.zeros(
            (X.shape[0], num_samples)
        )  # Initialize grid with the same number of features as X_train

        # Fill the grid with the selected dimensions
        grid[feature_x, :] = xx.ravel()  # Assign x values to the correct dimension
        grid[feature_y, :] = yy.ravel()  # Assign y values to the correct dimension

        # Generate random values for the non-selected dimensions
        for i in range(X.shape[0]):
            if i != feature_x and i != feature_y:
                grid[i, :] = np.random.rand(num_samples)  # Fill with random values

        # Generate predictions for the grid points
        grid_predictions = (
            self.predict(grid) >= 0.5 if not proba else self.predict(grid)
        )
        Z = grid_predictions.reshape(xx.shape)

        if transparency:
            plt.scatter(
                X[feature_x, :],
                X[feature_y, :],
                c=self.y_train.flatten(),
                edgecolors="k",
                marker="o",
                cmap=cmap,
                alpha=1,
            )
            if colorbar:
                plt.colorbar()
            plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap, levels=levels)
            if colorbar:
                plt.colorbar()
        else:
            plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap, levels=levels)
            if colorbar:
                plt.colorbar()
            plt.scatter(
                X[feature_x, :],
                X[feature_y, :],
                c=self.y_train.flatten(),
                edgecolors="k",
                marker="o",
                cmap=cmap,
            )
            if colorbar:
                plt.colorbar()
        plt.title("Predictions")
        plt.xlabel(f"Feature {feature_x + 1}")  # Dynamic labeling
        plt.ylabel(f"Feature {feature_y + 1}")  # Dynamic labeling
        plt.show()

    @staticmethod
    def metrics_histogram(*args, exclude: list = None):
        if exclude is None:
            exclude = []

        for arg in args:
            if not isinstance(arg, dict):
                raise ValueError("Args must be a dict type.")

        metrics = {k: [] for k, _ in args[0].items() if not k in exclude}

        for d in args:
            for key, value in d.items():
                if not key in exclude:
                    metrics[key].append(value)

        n_dicts = len(args)
        colors = [
            "#073642",
            "#839496",
            "#b58900",
            "#cb4b16",
            "#dc322f",
            "#d33682",
            "#6c71c4",
            "#268bd2",
            "#2aa198",
            "#859900",
        ]
        fig, axes = plt.subplots(4, 4, figsize=(20, 14))

        for i, (key, values) in enumerate(metrics.items()):
            ax = axes[i // 4, i % 4]
            bar_width = 0.8
            index = np.arange(n_dicts)

            for j, value in enumerate(values):
                ax.bar(index[j], value, bar_width, color=colors[j], label=f"DNN{j+1}")

            ax.set_title(key)
            ax.set_xticks([])

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.1, 0.5))
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.show()
