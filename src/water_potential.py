import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
import joblib

# ===================================================================
# Module 4: Water Potential & Survival Prediction (Scikit-Learn)
# Implements the mathematical models from Section 6.5 of the Report
# ===================================================================

def generate_synthetic_calibration_data(n_samples=200):
    """
    Generates synthetic calibration data based on the parameters
    described in the report since we don't have physical pressure chamber data.
    - theta goes from 90 (flat) down to ~20 (fully folded).
    - w_ratio goes from 1.0 down to ~0.4.
    - Psi_leaf goes from -0.5 MPa down to -4.5 MPa.
    - Survival is binary (0 or 1).
    """
    print(f"Generating {n_samples} synthetic physiological data points...")
    np.random.seed(42)
    
    # Generate random physiological states (flat vs folded)
    # Theta (Angle) between 20 and 90 degrees
    theta_data = np.random.uniform(20, 90, n_samples)
    
    # Width ratio (w) is highly correlated with theta
    w_data = 0.3 + (theta_data / 90.0) * 0.7 + np.random.normal(0, 0.05, n_samples)
    w_data = np.clip(w_data, 0.2, 1.0)
    
    # Psi_leaf = alpha * theta + beta * w + gamma + error
    # We construct it so that high theta (90, flat) -> Psi_leaf ~ -0.5 (healthy)
    # and low theta (20, folded) -> Psi_leaf ~ -4.0 (severe drought)
    psi_leaf_true = 0.04 * theta_data + 1.5 * w_data - 5.5 + np.random.normal(0, 0.2, n_samples)
    
    # Survival probability depends strongly on Psi_leaf
    # If Psi_leaf drops below -3.5, survival drops significantly
    z = 2.0 + 1.2 * psi_leaf_true
    survival_prob = 1 / (1 + np.exp(-z))
    survival_data = np.random.binomial(1, survival_prob)
    
    # Pack into feature matrix X: [Theta, W_ratio]
    X_features = np.column_stack((theta_data, w_data))
    
    return X_features, psi_leaf_true, survival_data

def train_water_potential_model(X, y_psi):
    """
    Trains a Linear Regression model:
    Psi_leaf(predicted) = α * θ + β * w + γ
    """
    print("Training Water Potential Linear Regression Model...")
    model_psi = LinearRegression()
    model_psi.fit(X, y_psi)
    
    preds = model_psi.predict(X)
    rmse = np.sqrt(mean_squared_error(y_psi, preds))
    
    print(f"Linear Eq: Psi_leaf = {model_psi.coef_[0]:.4f}*theta + {model_psi.coef_[1]:.4f}*w + {model_psi.intercept_:.4f}")
    print(f"Model RMSE: {rmse:.3f} MPa")
    return model_psi

def train_survival_model(y_psi, y_survival):
    """
    Trains a Logistic Regression model:
    P(survival) = 1 / (1 + exp(-(δ₀ + δ₁ * Ψleaf)))
    Note: Requires Psi_leaf as input feature.
    """
    print("Training Survival Probability Logistic Regression Model...")
    model_survival = LogisticRegression()
    y_psi_reshaped = y_psi.reshape(-1, 1)
    model_survival.fit(y_psi_reshaped, y_survival)
    
    print(f"Logistic Eq Params: δ₀ (Intercept) = {model_survival.intercept_[0]:.4f}, δ₁ (Coef) = {model_survival.coef_[0][0]:.4f}")
    return model_survival

def evaluate_and_plot(model_psi, model_survival, X, y_psi_true):
    """Visualizes the predictions."""
    y_psi_pred = model_psi.predict(X)
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: True vs Predicted Psi
    plt.subplot(1, 2, 1)
    plt.scatter(y_psi_true, y_psi_pred, alpha=0.5, color='blue')
    plt.plot([-5, 0], [-5, 0], 'r--')
    plt.xlabel('True $\Psi_{leaf}$ (MPa)')
    plt.ylabel('Predicted $\Psi_{leaf}$ (MPa)')
    plt.title('Water Potential Prediction')
    
    # Plot 2: Survival Probability Curve
    psi_range = np.linspace(-5, 0, 100).reshape(-1, 1)
    prob = model_survival.predict_proba(psi_range)[:, 1]
    
    plt.subplot(1, 2, 2)
    plt.plot(psi_range, prob, color='green')
    plt.xlabel('$\Psi_{leaf}$ (MPa)')
    plt.ylabel('Probability of Survival')
    plt.title('Survival Classification Model')
    
    plt.tight_layout()
    plt.savefig('data/simulated_outputs/water_potential_model.png')
    print("Saved data/simulated_outputs/water_potential_model.png")

if __name__ == "__main__":
    X_train, y_psi_train, y_survival_train = generate_synthetic_calibration_data()
    
    # Train Models
    psi_regressor = train_water_potential_model(X_train, y_psi_train)
    survival_classifier = train_survival_model(y_psi_train, y_survival_train)
    
    # Save the models
    joblib.dump(psi_regressor, 'models/regression/psi_model.pkl')
    joblib.dump(survival_classifier, 'models/regression/surv_model.pkl')
    print("Exported models/regression/psi_model.pkl and models/regression/surv_model.pkl")
    
    evaluate_and_plot(psi_regressor, survival_classifier, X_train, y_psi_train)
    
    print("\n--- Example Inference ---")
    # Simulate a severely folded leaf observation
    sample_theta = 25.0
    sample_w = 0.45
    sample_feature = np.array([[sample_theta, sample_w]])
    
    pred_psi = psi_regressor.predict(sample_feature)[0]
    pred_surv = survival_classifier.predict_proba([[pred_psi]])[0][1]
    
    print(f"Observation: Theta = {sample_theta} deg, w_ratio = {sample_w}")
    print(f"Predicted Leaf Water Potential: {pred_psi:.2f} MPa")
    print(f"Predicted Survival Probability: {pred_surv*100:.1f}%")
