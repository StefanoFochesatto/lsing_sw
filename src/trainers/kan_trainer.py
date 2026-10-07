import copy
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class KANTrainer:
    """
    Handles fast, analytic training of MonotonicKANMap models for transport map learning.
    Eliminates numerical quadrature and nested autodiff during training.
    """

    def __init__(
        self,
        base_map,
        num_features,
        hidden_dim=4,
        grid_size=8,
        spline_order=3,
        penalty_type="functional",
        learning_rate=0.01,
        max_epochs=100,
        patience=10,
        dev="cpu",
    ):
        """
        Args:
            base_map (list): List of MonotonicKANMap models (one per feature).
            num_features (int): Dimensionality d of dataset.
            hidden_dim (int): Summation terms Q.
            grid_size (int): Number of intervals in B-spline grid.
            spline_order (int): Spline degree.
            penalty_type (str): 'functional', 'coeff_group', or 'target_jacobian'.
            learning_rate (float): Adam optimizer learning rate.
            max_epochs (int): Maximum training epochs.
            patience (int): Early stopping patience.
            dev (str): Target device ('cpu' or 'cuda').
        """
        self.base_map = base_map
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.penalty_type = penalty_type
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = torch.device(dev)

    def train_model(
        self,
        training_data,
        target_index,
        other_indices,
        regularizations,
        validation_data,
        verbose=False,
    ):
        """
        Trains a MonotonicKANMap for target_index across candidate regularization values.

        Returns:
            optimal_model: Best model based on validation NLL.
            optimal_reg: Selected regularization coefficient.
            models_per_reg: Dict mapping reg_value to list of models.
            training_stats: Dict of timing and loss convergence diagnostics.
        """
        training_data = training_data.to(self.device)
        validation_data = validation_data.to(self.device)

        best_val_nll = float("inf")
        optimal_regularization = regularizations[0]
        optimal_model = None
        models_per_regularization = {}

        stats = {
            "reg_values": [],
            "val_nll": [],
            "epochs_trained": [],
            "time_per_reg": [],
        }

        desc = f"Training KAN feature {target_index}"
        reg_iter = tqdm(regularizations, desc=desc, leave=False) if verbose else regularizations

        for reg_value in reg_iter:
            t0 = time.perf_counter()
            model = copy.deepcopy(self.base_map[target_index]).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            early_stop_counter = 0
            best_epoch_val_nll = float("inf")
            best_model_weights = copy.deepcopy(model.state_dict())

            epochs_done = 0
            for epoch in range(self.max_epochs):
                model.train()
                # Fast closed-form analytic loss computation
                loss, nll, reg_pen = model.compute_loss(training_data, reg_value=reg_value)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epochs_done += 1

                # Validation unregularized negative log-likelihood (Section 5 / Eq 4 in paper)
                model.eval()
                with torch.no_grad():
                    _, val_nll, _ = model.compute_loss(validation_data, reg_value=0.0)
                    val_nll_val = val_nll.item()

                if val_nll_val < best_epoch_val_nll:
                    best_epoch_val_nll = val_nll_val
                    best_model_weights = copy.deepcopy(model.state_dict())
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

                if early_stop_counter >= self.patience:
                    break

            reg_time = time.perf_counter() - t0
            # Load best epoch weights for this reg value
            model.load_state_dict(best_model_weights)
            models_per_regularization.setdefault(reg_value, []).append(model)

            stats["reg_values"].append(reg_value)
            stats["val_nll"].append(best_epoch_val_nll)
            stats["epochs_trained"].append(epochs_done)
            stats["time_per_reg"].append(reg_time)

            if best_epoch_val_nll < best_val_nll:
                best_val_nll = best_epoch_val_nll
                optimal_regularization = reg_value
                optimal_model = copy.deepcopy(model)

        if optimal_model is None:
            optimal_model = copy.deepcopy(self.base_map[target_index])

        return optimal_model, optimal_regularization, models_per_regularization, stats


class KANPrecisionComputer:
    """
    Computes generalized precision matrix Omega for trained MonotonicKANMap models
    using closed-form analytic derivatives.
    """

    def __init__(self, trained_models, test_data, num_features):
        self.trained_models = trained_models
        self.test_data = test_data
        self.num_features = num_features

    def compute(self, split=False):
        """
        Computes precision matrix. If split=True, sums terms separately (|A| + |B|).
        """
        precision_matrix = []
        for feature_idx, model in enumerate(self.trained_models):
            if hasattr(model, "compute_precision_row"):
                if split:
                    row = model.compute_precision_row_split(self.test_data)
                else:
                    row = model.compute_precision_row(self.test_data)
            else:
                raise ValueError("Model must be an instance of MonotonicKANMap")
            precision_matrix.append(row)
        return np.array(precision_matrix)

    @staticmethod
    def make_symmetric(precision_matrix):
        transpose_matrix = precision_matrix.T
        symmetric_matrix = (transpose_matrix + precision_matrix) / 2.0
        max_val = np.max(np.abs(symmetric_matrix))
        if max_val > 0:
            normalized_matrix = symmetric_matrix / max_val
        else:
            normalized_matrix = symmetric_matrix
        np.fill_diagonal(normalized_matrix, 1.0)
        return normalized_matrix
