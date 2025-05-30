import pandas as pd
from ensemble_model import EnsembleModel
from price_optimizer_gurobi import PriceOptimizer

# === Step 1: Load Task Data ===
# Simulated input
# Each task has: complexity, urgency, market_demand, task_type, etc.
data = pd.read_csv("task_data.csv")  # Placeholder path

# === Step 2: Feature Processing ===
features = data[["complexity", "urgency", "market_demand"]]

# === Step 3: Load & Predict with Ensemble Model ===
model = EnsembleModel()
# In production, load pre-trained models instead of re-training
model.fit(features, data["true_duration"], data["true_compute"])
pred_duration, pred_compute = model.predict(features)

data["pred_duration"] = pred_duration
data["pred_compute"] = pred_compute

# Add prediction confidence if needed (e.g., based on residual std)
data["confidence"] = 1 - abs(data["pred_duration"] - data["true_duration"]) / (data["true_duration"] + 1e-6)

# === Step 4: Build Optimization Input ===
task_list = []
for _, row in data.iterrows():
    task_list.append({
        "id": row["task_id"],
        "base_cost": row["base_cost"],
        "duration": row["pred_duration"],
        "compute": row["pred_compute"],
        "demand_score": row["market_demand"],
        "urgency": row["urgency"],
        "confidence": row["confidence"]
    })

# === Step 5: Optimize Prices ===
optimizer = PriceOptimizer(min_margin=0.15, risk_weight=0.1)
agent_capacity = 1000  # Total compute units available
final_prices = optimizer.optimize_prices(task_list, agent_capacity)

# === Step 6: Output Results ===
for tid, price in final_prices.items():
    print(f"Task {tid}: Suggested Price = ${price:.2f}")
