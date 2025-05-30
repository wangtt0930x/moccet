import streamlit as st
import pandas as pd
from ensemble_model import EnsembleModel
from price_optimizer_gurobi import PriceOptimizer

st.set_page_config(page_title="AI Task Dynamic Pricing", layout="centered")
st.title("🚀 AI Agent Dynamic Pricing Engine")

st.markdown("""
This tool estimates completion time and compute resources for AI tasks,
then dynamically prices them using optimization logic.
""")

uploaded_file = st.file_uploader("📤 Upload Task Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Raw Data Preview", data.head())

    features = data[["complexity", "urgency", "market_demand"]]

    model = EnsembleModel()
    model.fit(features, data["true_duration"], data["true_compute"])
    pred_duration, pred_compute = model.predict(features)

    data["pred_duration"] = pred_duration
    data["pred_compute"] = pred_compute
    data["confidence"] = 1 - abs(pred_duration - data["true_duration"]) / (data["true_duration"] + 1e-6)

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

    agent_capacity = st.slider("Agent Total Compute Capacity", 100, 5000, 1000)

    optimizer = PriceOptimizer()
    prices = optimizer.optimize_prices(task_list, agent_capacity)

    st.write("### 💰 Optimized Prices:")
    result_df = pd.DataFrame.from_dict(prices, orient='index', columns=["Suggested Price"])
    try:
        result_df["Suggested Price"] = result_df["Suggested Price"].astype(float)
        st.dataframe(result_df.style.format("${:.2f}"))
    except:
        st.dataframe(result_df)
    result_df.index.name = "Task ID"

    st.download_button("📥 Download Price Table", result_df.to_csv().encode(), file_name="optimized_prices.csv", mime="text/csv")
