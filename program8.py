# Step 1: Install hmmlearn if you haven't already
# pip install hmmlearn

import numpy as np
from hmmlearn.hmm import GaussianHMM

# Example: Generating synthetic sequential data
np.random.seed(42)
X = np.concatenate([np.random.normal(0, 1, (100, 1)), np.random.normal(5, 1, (100, 1))])

# Reshaping the data as the model expects a 2D array: (n_samples, n_features)
X = X.reshape(-1, 1)

# Step 2: Create an HMM model
model = GaussianHMM(n_components=2, covariance_type="diag", random_state=42)

# Step 3: Fit the model to the data (no need for n_trials)
model.fit(X)

# Step 4: Predict the hidden states
hidden_states = model.predict(X)

# Print the predicted hidden states
print(f"Predicted hidden states: {hidden_states[:10]}")

# Step 5: Optionally, you can visualize the hidden states and observations
import matplotlib.pyplot as plt
plt.plot(X, label="Observations")
plt.plot(hidden_states, label="Predicted Hidden States", linestyle="--")
plt.legend()
plt.show()
