import os

import joblib
import pandas as pd
import streamlit as st

MODEL_FILENAME = "insurance_model.joblib"
POLICY_TYPES = ["Gold", "Silver", "Bronze"]

DUMMY_CLAIMS = [
    {"Customer_Age": 25, "Claim_Amount": 1200.0, "Past_Claims_Count": 0, "Policy_Type": "Silver"},
    {"Customer_Age": 34, "Claim_Amount": 5400.0, "Past_Claims_Count": 1, "Policy_Type": "Gold"},
    {"Customer_Age": 47, "Claim_Amount": 8900.0, "Past_Claims_Count": 2, "Policy_Type": "Bronze"},
    {"Customer_Age": 55, "Claim_Amount": 2300.0, "Past_Claims_Count": 0, "Policy_Type": "Silver"},
    {"Customer_Age": 62, "Claim_Amount": 15000.0, "Past_Claims_Count": 3, "Policy_Type": "Gold"},
    {"Customer_Age": 28, "Claim_Amount": 7200.0, "Past_Claims_Count": 1, "Policy_Type": "Bronze"},
    {"Customer_Age": 38, "Claim_Amount": 4600.0, "Past_Claims_Count": 0, "Policy_Type": "Gold"},
    {"Customer_Age": 49, "Claim_Amount": 9800.0, "Past_Claims_Count": 2, "Policy_Type": "Silver"},
    {"Customer_Age": 30, "Claim_Amount": 3100.0, "Past_Claims_Count": 1, "Policy_Type": "Bronze"},
    {"Customer_Age": 41, "Claim_Amount": 12500.0, "Past_Claims_Count": 2, "Policy_Type": "Gold"},
]


def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {MODEL_FILENAME}. Place it next to app.py.")
        return None

    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return None


def init_session_state():
    defaults = {
        "Customer_Age": 35,
        "Claim_Amount": 5000.0,
        "Past_Claims_Count": 0,
        "Policy_Type": "Silver",
        "prediction": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def prefill_from_claim(claim):
    st.session_state.Customer_Age = int(claim["Customer_Age"])
    st.session_state.Claim_Amount = float(claim["Claim_Amount"])
    st.session_state.Past_Claims_Count = int(claim["Past_Claims_Count"])
    st.session_state.Policy_Type = claim["Policy_Type"]
    st.session_state.prediction = None


def build_dummy_claims_df():
    return pd.DataFrame(DUMMY_CLAIMS).assign(Claim_Label=[f"Sample {i + 1}" for i in range(len(DUMMY_CLAIMS))])[
        ["Claim_Label", "Customer_Age", "Claim_Amount", "Past_Claims_Count", "Policy_Type"]
    ]


def main():
    st.set_page_config(page_title="Insurance Claim Predictor", layout="wide")
    st.title("Insurance Claim Status Predictor")
    st.write("Use the form below to predict whether a claim will be approved or denied.")

    init_session_state()
    model = load_model()

    sidebar = st.sidebar
    sidebar.header("10 Test Claims")
    dummy_df = build_dummy_claims_df()
    sidebar.table(dummy_df)

    for idx, claim in enumerate(DUMMY_CLAIMS):
        if sidebar.button(f"Load Sample {idx + 1}", key=f"load_claim_{idx}"):
            prefill_from_claim(claim)

    with st.container():
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Claim Input")
            age = st.number_input(
                "Customer Age",
                min_value=18,
                max_value=120,
                value=st.session_state.Customer_Age,
                key="Customer_Age",
            )
            claim_amount = st.number_input(
                "Claim Amount",
                min_value=0.0,
                max_value=100000.0,
                value=st.session_state.Claim_Amount,
                step=100.0,
                format="%.2f",
                key="Claim_Amount",
            )
            past_claims = st.number_input(
                "Past Claims Count",
                min_value=0,
                max_value=20,
                value=st.session_state.Past_Claims_Count,
                key="Past_Claims_Count",
            )
            policy_type = st.selectbox(
                "Policy Type",
                options=POLICY_TYPES,
                index=POLICY_TYPES.index(st.session_state.Policy_Type),
                key="Policy_Type",
            )

            st.write("")
            predict_button = st.button("Predict Claim Status")

            if predict_button:
                st.session_state.prediction = None
                if model is not None:
                    # 1. Prepare base data
                    input_data = {
                        "Customer_Age": age,
                        "Claim_Amount": claim_amount,
                        "Past_Claims_Count": past_claims,
                        "Policy_Type_Gold": 0,
                        "Policy_Type_Silver": 0,
                    }

                    # 2. Set the correct policy flag to 1
                    # Note: Using the exact names: Policy_Type_Gold and Policy_Type_Silver
                    if policy_type == "Gold":
                        input_data["Policy_Type_Gold"] = 1
                    elif policy_type == "Silver":
                        input_data["Policy_Type_Silver"] = 1

                    # 3. Create DataFrame and "Proactively" order columns
                    input_df = pd.DataFrame([input_data])
                    
                    # PROACTIVE CHECK: Force the DataFrame to match the model's expected feature order
                    # This prevents 'Feature names unseen' errors completely
                    try:
                        input_df = input_df[model.feature_names_in_]
                        
                        prediction = model.predict(input_df)[0]
                        st.session_state.prediction = "Approved" if int(prediction) == 1 else "Denied"
                    except Exception as exc:
                        st.error(f"Prediction logic error: {exc}")
                        st.write("Model expected these features:", model.feature_names_in_)
                        st.write("Your input features:", input_df.columns.tolist())
                else:
                    st.warning("Unable to predict because the model could not be loaded.")

        with col2:
            st.subheader("Prediction Result")
            if st.session_state.prediction is not None:
                if st.session_state.prediction == "Approved":
                    st.success("Approved")
                else:
                    st.error("Denied")
            else:
                st.info("Enter claim details and click Predict Claim Status.")

    st.markdown("---")
    st.subheader("How to use")
    st.write(
        "Select a sample claim from the sidebar to auto-fill the form, then click `Predict Claim Status`."
    )


if __name__ == "__main__":
    main()
