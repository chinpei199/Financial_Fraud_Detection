
# cd C:\Users\User\Downloads
# streamlit run principleDS.py
#pip install joblib
#pip install scikit-learn
#pip install scikit-learn==1.6.1






import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# Define fixed type mapping
type_mapping = {'PAYMENT':3, 'TRANSFER':4, 'CASH_OUT':1, 'DEBIT':2, 'CASH_IN':5}





#  trained model
try:
    model = load(r"C:\Users\User\Downloads\xgfraud_detection_model.joblib")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# title
st.title("🔍 Advanced Fraud Detection System")
st.markdown("""
Upload your transaction data, select a record, and get real-time fraud predictions.
""")

# to upload file 
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, dtype={'type': str}, low_memory=False)
    
    # Clean empty records
    initial_count = len(df)
    df_clean = df.dropna().copy()
    cleaned_count = len(df_clean)
    

    st.success(f"{cleaned_count} records detected")

    
    # Display cleaned data with expandable section
    with st.expander("View Cleaned Transaction Data", expanded=False):
        st.dataframe(df_clean.style.highlight_null(props="color: red;"), height=300)

    # --- 2. Interactive Record Selection ---
    st.subheader("🔎 Transaction Inspector")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Enhanced record selector with search
        selected_index = st.selectbox(
            "Select transaction:",
            options=range(len(df_clean)),
            format_func=lambda x: (
                f"TxID: {df_clean.iloc[x].get('transactionID', x)} | "
                f"Amount: ${df_clean.iloc[x]['amount']:.2f} | "
                f"Type: {df_clean.iloc[x]['type']}"
            ),
            key="record_selector"
        )
        
        # Add manual refresh button
        st.caption("Can't find your transaction? Try reordering:")
        sort_by = st.radio("Sort by:", ["Amount (High→Low)", "Amount (Low→High)", "Original Order"])
        
        if sort_by == "Amount (High→Low)":
            df_clean = df_clean.sort_values("amount", ascending=False)
        elif sort_by == "Amount (Low→High)":
            df_clean = df_clean.sort_values("amount", ascending=True)
    
    with col2:
        # Display selected record details
        selected_row = df_clean.iloc[selected_index]
        st.subheader("Transaction Details")
        
        # Create a metrics view
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Amount", f"${selected_row['amount']:,.2f}")
        metric_col2.metric("Type", selected_row['type'])
        
        # Show additional details in JSON format
        with st.expander("View all fields"):
            st.json({k: v for k, v in selected_row.items() if pd.notna(v)})

    # --- 3. Prediction Section ---
    st.subheader("🧠 Fraud Prediction Analysis")
    
    # Prepare raw features
    raw_type = selected_row['type']
    raw_amount = selected_row['amount']
    raw_step = selected_row.get('step', 1)  # Default to 1 if step doesn't exist
    raw_isFlaggedFraud = selected_row.get('isFlaggedFraud', 0)  # Default to 0 if doesn't exist

    # Encode type using fixed mapping
    try:
        type_encoded = type_mapping.get(raw_type)
        if type_encoded is None:
            raise ValueError(f"Unknown transaction type: '{raw_type}'")
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Build feature DataFrame (adjust columns to match your model's expected features)
    features = pd.DataFrame([[raw_step, raw_amount, raw_isFlaggedFraud, type_encoded]], 
                          columns=['step', 'amount', 'isFlaggedFraud', 'type_encoded'])
    
    # Prediction button with loading state
    if st.button("Run Fraud Prediction", type="primary"):
        with st.spinner("Analyzing transaction..."):
            try:
                # Make prediction
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0][1]
                
                # Display results
                st.subheader("Results")
                
                # Visual risk indicator
                risk_color = "#ff4b4b" if prediction == 1 else "#2ecc71"
                st.markdown(f"""
                <div style="background-color:{risk_color}20; padding:15px; border-radius:10px; border-left:5px solid {risk_color}">
                    <h3 style="color:{risk_color}; margin-top:0;">{"🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE TRANSACTION"}</h3>
                    <p>Confidence: <b>{proba:.1%}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability meter
                st.progress(float(proba))
                st.caption(f"Fraud Risk Score: {proba:.1%}")
                
                # Explanation section
                with st.expander("How to interpret these results"):
                    st.markdown("""
                    - **<95% confidence**: Very likely legitimate
                    - **95-99% confidence**: Suspicious
                    - **>99% confidence**: Very likely fraudulent
                    """)
                
                # Show model features importance (if available)
                try:
                    if hasattr(model, 'feature_importances_'):
                        # st.subheader("Key Factors in This Prediction")
                        importances = pd.DataFrame({
                            'Feature': model.feature_names_in_,
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        st.bar_chart(importances.set_index('Feature'))
                except Exception:
                    pass
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- Sidebar with additional info ---
with st.sidebar:
    st.markdown("""
    ### How to Use
    1. Upload your transaction CSV
    2. Select a transaction
    3. Click "Run Fraud Prediction"
    
    ### Supported Transaction Types
    - PAYMENT 
    - TRANSFER 
    - CASH_OUT 
    - DEBIT 
    - CASH_IN 
    """)
    
    # # Add model info
    # st.markdown("""
    # ---
    # **Model Information**  
    # Using fixed type encoding mapping
    # """)
























