class RuleBasedAgent:
        """
        Initializes the rule-based agent.

        Args:
            config: The environment configuration, which contains reward curves per container.
            vol_margin: Volume threshold (in absolute units) below the reward peak
                        at which the container is considered for emptying.
        """
    def __init__(self, config: EnvConfig, vol_margin=1):
        self.vol_margin = vol_margin # Acceptable deviation from reward peak
        self.enabled_containers = config.enabled_containers # Use the first reward peak as the target "optimal" volume per container
        self.best_volumes = [
            config.containers[cid].reward.peaks[0]
            for cid in self.enabled_containers
        ]

    def predict(self, obs, deterministic=True):
        """
        Predicts the container to empty based on proximity to peak volume.

        Args:
            obs: A dictionary with key "volume" containing a NumPy array of volumes.
            deterministic: Placeholder for interface compatibility.

        Returns:
            An action index (container to empty) or a special "wait" action.
        """
        differences = np.array(self.best_volumes) - np.array(obs["volume"])
        candidate_containers = np.where(differences <= self.vol_margin)[0]
        if len(candidate_containers) == 0:
            return len(self.enabled_containers), 0  # Wait if none are within the margin
        return candidate_containers[0], 0 # Return the first eligible container
