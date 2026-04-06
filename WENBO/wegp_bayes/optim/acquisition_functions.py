import torch
import numpy as np
from scipy.stats import norm


class EI_NUTS:
    def __init__(self, model, best_f):
        """
        Initialize the EI object.

        Args:
        - model: The GP model that provides the predictions (mean and std).
        - best_f: The current best (minimum) value observed in the optimization process.
        """
        self.model = model
        self.best_f = best_f

    def evaluate(self, x, num_model_samples=128):
        """
        Evaluate the Expected Improvement (EI) at a given point `x`.
        
        Args:
        - x: Candidate points for evaluation (could be a batch of points).
        - num_model_samples: The number of posterior samples used for the EI computation.

        Returns:
        - The expected improvement for the given points `x`.
        """
        # x = torch.tensor(x, dtype=torch.float64)
        # print(f"x shape: {x.shape}")
        # print(f"x: {x}")
        self.model.eval() 

        ei_per_sample = []
        with torch.no_grad():
            mu, sigma = self.model.predict(x, return_std=True)

            # print(f"mu: {mu}")
            # print(f"sigma: {sigma}")
            # print(f"mu shape: {mu.shape}")
            # print(f"sigma shape: {sigma.shape}")
            # 确保 mu 和 sigma 有正确的维度
            if len(mu.shape) == 1:
                mu = mu.unsqueeze(0)
            if len(sigma.shape) == 1:
                sigma = sigma.unsqueeze(0)
            # print(f"mu shape: {mu.shape}")
            # print(f"sigma shape: {sigma.shape}")
            for i in range(num_model_samples):
                mu_i = mu[i].numpy()   
                sigma_i = sigma[i].numpy()  
                
                Z = (mu_i - self.best_f) / sigma_i
                ei = (mu_i - self.best_f) * norm.cdf(Z) + sigma_i * norm.pdf(Z)
                ei_per_sample.append(ei)
        avg_ei = np.mean(ei_per_sample, axis=0)
        # print(f"avg_ei shape: {avg_ei.shape}")
        # print(f"avg_ei: {avg_ei}")
        return avg_ei

class EI:
    def __init__(self, model, best_f):
        self.model = model
        self.best_f = best_f

    def evaluate(self, x):
        self.model.eval()
        with torch.no_grad():
            mu, sigma = self.model.predict(x, return_std=True)
        Z = (mu.numpy() - self.best_f) / sigma.numpy()
        return (mu.numpy() - self.best_f) * norm.cdf(Z) + sigma.numpy() * norm.pdf(Z)