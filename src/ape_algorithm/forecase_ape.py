# ForecastAPEAgent with Kalman filters

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

class KalmanFilter:
        """
        A 1D Kalman Filter to track volume and fill rate for a single container.
        State vector x = [volume, fill_rate]^T.
        """
    def __init__(self, initial_volume, initial_rate=0.0, Q=None, R=None):
        self.x = np.array([[initial_volume], [initial_rate]])  # state: [volume, fill_rate] # Initial state estimate
        self.P = np.eye(2) # Initial state covariance
        self.A = np.array([[1, 1], [0, 1]])  # State transition matrix (volume += rate)
        self.H = np.array([[1, 0]])  # Observation model (only observe volume)
        self.Q = Q if Q is not None else np.diag([1e-2, 1e-4]) # Process noise covariance
        self.R = R if R is not None else np.array([[1e-2]]) # Measurement noise covariance
#Predict next state and covariance based on system dynamics.
    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x.flatten() # Returns (estimated_volume, estimated_rate)
#Update state estimate based on new observation z.
    def update(self, z):
        y = np.array([[z]]) - self.H @ self.x # Measurement residual
        S = self.H @ self.P @ self.H.T + self.R # Residual covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain
        self.x = self.x + K @ y # Updated state estimate
        self.P = (np.eye(2) - K @ self.H) @ self.P # Updated covariance

class ForecastAPEAgent:
        """
        ForecastAPEAgent:
        - Uses per-container Kalman filters to estimate current volume and fill rate.
        - Forecasts volume k steps into the future.
        - Computes a priority score combining reward proximity and overflow risk.
        """
    def __init__(self, container_ids, reward_params, vmax_map, lambda_=0.2, k=3):
        self.filters = {cid: KalmanFilter(initial_volume=0.0) for cid in container_ids}
        self.reward_params = reward_params
        self.vmax_map = vmax_map
        self.lambda_ = lambda_
        self.k = k

    def compute_priority(self, cid):
        """
        Compute priority π_{i,t} for container `cid`:
        π_{i,t} = prox_{i,t} - λ * ρ_{i,t}
        where:
        - prox: reward proximity score
        - ρ: overflow probability based on future forecast
        """
        v_hat, alpha_hat = self.filters[cid].predict()
        v_future = v_hat + self.k * alpha_hat

        peaks = self.reward_params[cid]["peaks"]
        widths = self.reward_params[cid]["widths"]
        heights = self.reward_params[cid]["heights"]

        prox = max([
            h * np.exp(-((v_future - peak) ** 2) / (2 * w ** 2))
            for peak, w, h in zip(peaks, widths, heights)
        ])

        vmax = self.vmax_map[cid]
        rho = 1 - norm.cdf(vmax, loc=v_future, scale=np.sqrt(self.filters[cid].P[0, 0]))
        return prox - self.lambda_ * rho

    def update_filter(self, cid, observed_volume):
        """
        Incorporate observed volume into Kalman filter for the given container.
        """
        self.filters[cid].update(observed_volume)

    def get_priorities(self, observation_dict):
        """
        Compute priority scores for all containers based on current observations.
        observation_dict: Dict mapping cid -> observed volume
        Returns:
        - Dictionary mapping cid -> priority score
        """
        priorities = {}
        for cid, obs in observation_dict.items():
            self.update_filter(cid, obs)
            priorities[cid] = self.compute_priority(cid)
        return priorities
