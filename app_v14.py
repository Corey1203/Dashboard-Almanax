import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.feature_extraction.text import CountVectorizer

# Feature name mapping
feature_mapping = {
    "wallet_access_status": "Wallet Access",
    "pii_handling_status": "PII Handling",
    "spending_limits": "Spending Limits",
    "SLOC": "Source Lines of Code",
    "LLOC": "Logical Lines of Code",
    "CLOC": "Comment Lines of Code",
    "NF": "Number of Functions",
    "WMC": "Weighted Methods per Class",
    "NL": "Number of Loops",
    "NLE": "Number of Loop Exits",
    "NUMPAR": "Number of Parameters",
    "NOS": "Number of Statements",
    "DIT": "Depth of Inheritance Tree",
    "NOA": "Number of Attributes",
    "NOD": "Number of Descendants",
    "CBO": "Coupling Between Objects",
    "NA": "Number of Associations",
    "NOI": "Number of Inheritance",
    "Avg_McCC": "Avg Cyclomatic Complexity",
    "Avg_NL": "Avg Loop Count",
    "Avg_NLE": "Avg Loop Exits",
    "Avg_NUMPAR": "Avg Parameter Count",
    "Avg_NOS": "Avg Statement Count",
    "Avg_NOI": "Avg Inheritance Count"
}

# === Load Data ===
df = pd.read_excel("AI_Agent_Records_Final_10327.xlsx")
base_time = pd.Timestamp("2025-01-01")
df['timestamp'] = [base_time + timedelta(hours=i) for i in range(len(df))]

loss_tabnet = pd.read_csv(r"model comparison/tabnet_loss_curve.csv")
loss_tabnet_pca = pd.read_csv(r"model comparison/tabnet_pca_loss_curve.csv")
loss_tensorboard = pd.read_csv(r"model comparison/tensorboard_loss_curve.csv")
pred_comparison = pd.read_csv(r"model comparison/model_prediction_comparison.csv")
pred_comparison_pca = pd.read_csv(r"model comparison/model_prediction_comparison_pca.csv")
actual_vs_pred = pd.read_csv(r"model comparison/actual_vs_predicted_risk_score.csv")

# Derive risk_level from risk_score
def assign_risk_level(score):
    if score < 35:
        return "Low"
    elif score < 70:
        return "Medium"
    else:
        return "High"

df['risk_level'] = df['risk_score'].apply(assign_risk_level)

# === Load Model and SHAP Explainer ===
xgb_model = joblib.load("xgboost_model.pkl")
import shap
import joblib

# Load raw XGBoost model directly
xgb_model = joblib.load("xgboost_model.pkl")

# Create SHAP explainer from model

# Safely extract XGBRegressor from pipeline
if isinstance(xgb_model, xgboost.XGBRegressor):
    model_to_explain = xgb_model
elif hasattr(xgb_model, 'named_steps'):
    print("Pipeline steps:", xgb_model.named_steps)  # Debug
    model_to_explain = list(xgb_model.named_steps.values())[-1]
    if not isinstance(model_to_explain, xgboost.XGBRegressor):
        raise TypeError(f"Unsupported model type: {type(model_to_explain)}")
else:
    raise TypeError(f"Unsupported object type loaded: {type(xgb_model)}")

explainer = shap.TreeExplainer(model_to_explain)


X = df[[
    'wallet_access_status', 'pii_handling_status', 'spending_limits',
    'SLOC', 'LLOC', 'CLOC', 'NF', 'WMC', 'NL', 'NLE', 'NUMPAR', 'NOS',
    'DIT', 'NOA', 'NOD', 'CBO', 'NA', 'NOI',
    'Avg_McCC', 'Avg_NL', 'Avg_NLE', 'Avg_NUMPAR', 'Avg_NOS', 'Avg_NOI'
]]
agent_ids = df["AI Agent"].values

# === Sidebar Filters ===
st.sidebar.title("🔍 Navigation")

st.sidebar.markdown("### 🧩 Toggle Dashboard Sections")
show_overview = st.sidebar.checkbox("Show Dashboard Overview", value=True, key="show_overview")
show_eda = st.sidebar.checkbox("Show Agent Action EDA", value=True, key="show_eda")
show_shap = st.sidebar.checkbox("Show SHAP Explanation", value=True, key="show_shap")
show_compare = st.sidebar.checkbox("Show Model Comparison", value=True, key="show_compare")

tabs = []
tab_labels = []
if show_overview:
    tab_labels.append("📊 Overview")
if show_eda:
    tab_labels.append("📈 Action EDA")
if show_shap:
    tab_labels.append("🧠 SHAP Insights")
if show_compare:
    tab_labels.append("📉 Model Comparison")

tabs = st.tabs(tab_labels)

st.sidebar.header("🧮 Filter Agents")
risk_level_filter = st.sidebar.multiselect(
    "Select Risk Level(s)",
    options=df['risk_level'].unique(),
    default=df['risk_level'].unique()
)
start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [df['timestamp'].min(), df['timestamp'].max()]
)

filtered_df = df[
    (df['risk_level'].isin(risk_level_filter)) &
    (df['timestamp'] >= pd.to_datetime(start_date)) &
    (df['timestamp'] <= pd.to_datetime(end_date))
]

if show_overview:
    with tabs[tab_labels.index("📊 Overview")]:
        st.title("📊 AI Agent Risk Dashboard Overview")

        # Summary Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Agents", len(filtered_df))
        with col2:
            st.metric("Average Risk Score", round(filtered_df['risk_score'].mean(), 2))

        # Distribution of Risk Scores
        fig = px.histogram(filtered_df, x='risk_score', nbins=20, color='risk_level', 
                         color_discrete_sequence=['#00338D', '#005EB8', '#0072CE'],
                         title="Distribution of Risk Scores")
        st.plotly_chart(fig, use_container_width=True)

        # Risk Score Summary by Risk Level - Combined View
        st.subheader("📊 Risk Score Overview")
        
        # Calculate group statistics
        group_stats = filtered_df.groupby('risk_level')['risk_score'].agg(['count', 'mean', 'min', 'max']).reset_index()
        group_stats.columns = ['Risk Level', 'Count', 'Avg Score', 'Min Score', 'Max Score']
        
        # Create two columns for the combined view
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for counts
            fig_count = px.bar(group_stats, 
                              x='Risk Level', 
                              y='Count',
                              color='Risk Level',
                              color_discrete_sequence=['#00338D', '#005EB8', '#0072CE'],
                              title='Number of Agents by Risk Level',
                              text='Count')
            fig_count.update_traces(textposition='outside')
            fig_count.update_layout(showlegend=False)
            st.plotly_chart(fig_count, use_container_width=True)

        with col2:
            # Bar chart for average scores
            fig_avg = px.bar(group_stats, 
                            x='Risk Level', 
                            y='Avg Score',
                            color='Risk Level',
                            color_discrete_sequence=['#00338D', '#005EB8', '#0072CE'],
                            title='Average Risk Score by Risk Level',
                            text='Avg Score')
            fig_avg.update_traces(texttemplate='%{text:.1f}', textposition='outside')
            fig_avg.update_layout(showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)

        # Show detailed table in an expander
        with st.expander("📋 View Detailed Statistics"):
            st.dataframe(group_stats.style.format({'Avg Score': '{:.2f}'}))

        # Risk Score Trend Over Time
        st.subheader("📈 Risk Score Trend Over Time")
        trend_df = filtered_df.copy()
        trend_df['date'] = trend_df['timestamp'].dt.date
        trend_summary = trend_df.groupby(['date', 'risk_level'])['risk_score'].mean().reset_index()
        fig_trend = px.line(
            trend_summary,
            x='date',
            y='risk_score',
            color='risk_level',
            title='Average Risk Score Trend by Risk Level',
            markers=True,
            color_discrete_sequence=['#00338D', '#005EB8', '#0072CE']
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Enhanced Actionable Recommendations with equal height cards
        st.subheader("✅ Actionable Recommendations")
        
        rec_col1, rec_col2, rec_col3 = st.columns(3)
        
        # CSS for equal height cards
        st.markdown("""
        <style>
            .recommendation-card {
                height: 280px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                padding: 15px;
                border-radius: 10px;
                border-left: 5px solid;
                margin-bottom: 20px;
            }
            .recommendation-title {
                margin-top: 0;
                font-size: 1.2rem;
            }
            .recommendation-list {
                margin-bottom: 0;
                padding-left: 20px;
            }
            .recommendation-list li {
                margin-bottom: 8px;
            }
        </style>
        """, unsafe_allow_html=True)
        
        with rec_col1:
            st.markdown(f"""
            <div class="recommendation-card" style="background-color: #e6f7e6; border-left-color: #4CAF50;">
                <h3 class="recommendation-title" style="color: #4CAF50;">🟢 Low Risk</h3>
                <ul class="recommendation-list">
                    <li>Monitor them periodically</li>
                    <li>No immediate action needed</li>
                    <li>Standard surveillance</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col2:
            st.markdown(f"""
            <div class="recommendation-card" style="background-color: #fff8e6; border-left-color: #FFC107;">
                <h3 class="recommendation-title" style="color: #FFC107;">🟡 Medium Risk</h3>
                <ul class="recommendation-list">
                    <li>Weekly activity review</li>
                    <li>Check for abnormalities</li>
                    <li>Enhanced monitoring</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with rec_col3:
            st.markdown(f"""
            <div class="recommendation-card" style="background-color: #ffebee; border-left-color: #F44336;">
                <h3 class="recommendation-title" style="color: #F44336;">🔴 High Risk</h3>
                <ul class="recommendation-list">
                    <li>Immediate audit required</li>
                    <li>Restrict access immediately</li>
                    <li>Check network activity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Level Distribution (alternative view)
        st.subheader("📊 Risk Level Distribution")
        fig_pie = px.pie(filtered_df, 
                        names='risk_level', 
                        title='Proportion of Agents by Risk Level',
                        color='risk_level',
                        color_discrete_sequence=['#00338D', '#005EB8', '#0072CE'])
        st.plotly_chart(fig_pie, use_container_width=True)

if show_eda:
    with tabs[tab_labels.index("📈 Action EDA")]:
        st.title("🛠️ AI Agent Action EDA")

        st.subheader("Network Activity Analysis")
        fig_activity = px.histogram(df, x='number_of_network_activity', nbins=30, 
                                   marginal='box', title="Network Activity Histogram", 
                                   color_discrete_sequence=['#005EB8'])
        st.plotly_chart(fig_activity, use_container_width=True)

        st.subheader("Wallet Access Actions")
        df['wallet_access_actions'] = df['wallet_access_actions'].fillna('')
        vectorizer_wallet = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        wallet_matrix = vectorizer_wallet.fit_transform(df['wallet_access_actions'])
        wallet_df = pd.DataFrame(wallet_matrix.toarray(), columns=vectorizer_wallet.get_feature_names_out())
        wallet_counts = wallet_df.sum().sort_values(ascending=False).head(10)

        fig_wallet = px.bar(
            x=wallet_counts.values,
            y=wallet_counts.index,
            orientation='h',
            title="Top 10 Wallet Access Actions",
            labels={'x': 'Number of Agents', 'y': 'Action'},
            color_discrete_sequence=['#0072CE']
        )
        fig_wallet.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_wallet, use_container_width=True)

        st.subheader("PII Handling Actions")
        df['pii_handling_actions'] = df['pii_handling_actions'].fillna('')
        vectorizer_pii = CountVectorizer(token_pattern=r'(?u)\b\w+\b')
        pii_matrix = vectorizer_pii.fit_transform(df['pii_handling_actions'])
        pii_df = pd.DataFrame(pii_matrix.toarray(), columns=vectorizer_pii.get_feature_names_out())
        pii_counts = pii_df.sum().sort_values(ascending=False).head(5)

        fig_pii = px.bar(
            x=pii_counts.values,
            y=pii_counts.index,
            orientation='h',
            title="Top 5 PII Handling Actions",
            labels={'x': 'Number of Agents', 'y': 'Action'},
            color=pii_counts.values,
            color_continuous_scale='Reds'
        )
        fig_pii.update_layout(yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig_pii, use_container_width=True)

if show_shap:
    with tabs[tab_labels.index("🧠 SHAP Insights")]:
        st.title("🧠 SHAP Explanation")
        
        selected_agent = st.selectbox("Choose an Agent", agent_ids)
        agent_index = df[df["AI Agent"] == selected_agent].index[0]

        X_transformed = xgb_model.named_steps['preprocessor'].transform(X)
        shap_values = explainer(X_transformed)

        # Get feature names - use original column names if SHAP doesn't have them
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names is not None:
            feature_names = [feature_mapping.get(name, name) for name in shap_values.feature_names]
        else:
            feature_names = [feature_mapping.get(name, name) for name in X.columns]

        # Create waterfall plot
        st.subheader(f"SHAP Waterfall Plot for {selected_agent}")
        plt.figure(figsize=(10, 6))
        shap_val = shap_values[agent_index]
        
        # Create waterfall plot with proper feature names
        shap.plots.waterfall(shap_val, max_display=15, show=False)
        
        # Get the current axes and modify the y-tick labels
        ax = plt.gca()
        y_ticks = ax.get_yticks()
        # Only modify the labels that are within the range of our feature names
        y_ticks = [int(tick) for tick in y_ticks if tick < len(feature_names)]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([feature_names[int(tick)] for tick in y_ticks])
        
        plt.title(f"SHAP Explanation for {selected_agent}", pad=20)
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.close()

# Load Data
base_path = "model comparison/"
loss_tabnet = pd.read_csv(base_path + "tabnet_loss_curve.csv")
loss_tabnet_pca = pd.read_csv(base_path + "tabnet_pca_loss_curve.csv")
df_comparison = pd.read_csv(base_path + "model_comparison.csv")
df_comparison_pca = pd.read_csv(base_path + "model_comparison_pca.csv")
df_tabtransformer = pd.read_csv(base_path + "actual_vs_predicted_tabtransformer.csv")
df_fi_tabnet_pca = pd.read_csv(base_path + "feature_importance_tabnet_pca.csv")
df_fi_lgbm = pd.read_csv(base_path + "feature_importance_lgbm.csv")

mck_blue = "#00338D"

if 'tabs' in locals() and 'tab_labels' in locals():
    with tabs[tab_labels.index("📉 Model Comparison")]:
        st.title("📉 Model Comparison and Insights")

        tab1, tab2, tab3, tab4 = st.tabs([
            "Prediction Accuracy ->",
            "Training Dynamics ->",
            "TabTransformer Focus ->",
            "Feature Attribution"
        ])


        # 1. Model Prediction Accuracy
with tab1:
    st.header("Model Prediction Accuracy (Original vs. PCA)")
    st.markdown("""
    This comparison evaluates how well each model predicts the actual risk score, and how dimensionality reduction (via PCA) affects performance.
    """)

    melted_df = df_comparison_pca.melt(id_vars=["actual_jitter"], var_name="model", value_name="predicted_score")

    fig = px.scatter(melted_df, x="actual_jitter", y="predicted_score", color_discrete_sequence=["#005EB8", "#0072CE", "#00A3E0", "#002D72"],
                     labels={"actual_jitter": "Actual Risk Score", "predicted_score": "Predicted Risk Score"},
                     title="Prediction Accuracy: PCA vs. Original", template="plotly_white")
    fig.add_shape(type='line', x0=0, y0=0, x1=100, y1=100,
                  line=dict(dash='dash', color='black'))
    st.plotly_chart(fig, use_container_width=True)
    st.info("🟢 Insight: XGBoost Stacked PCA has the lowest RMSE, showing PCA can help performance in some models.")

# 2. Training Dynamics
with tab2:
    st.header("Training Dynamics Across Models")
    st.markdown("Understand how each model learns across epochs to assess stability and convergence.")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.line(loss_tabnet, x="epoch", y=["training_loss", "validation_rmse"],
                       title="TabNet Loss Curve (Original)", template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(loss_tabnet_pca, x="epoch", y=["training_loss", "validation_rmse"],
                       title="TabNet Loss Curve (PCA)", template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    st.info("🟡 Observation: PCA enables faster convergence and more stable validation for TabNet.")

# 3. TabTransformer Focus
with tab3:
    st.header("TabTransformer: Prediction Distribution")
    st.markdown("Zoom into TabTransformer's prediction behavior. It tends to underestimate high-risk agents.")
    fig3 = px.scatter(df_tabtransformer, x="actual", y="predicted", trendline="ols",
                      labels={"actual": "Actual Risk Score", "predicted": "Predicted Risk Score"},
                      template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)
    st.info("🔴 Finding: High variance at upper risk scores suggests limited generalization on complex cases.")

# 4. Feature Attribution
with tab4:
    st.header("Feature Importance Comparison")
    st.markdown("Examine what each model considered most relevant for predicting risk.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("TabNet (PCA)")
        fig_fi_tabnet = px.bar(df_fi_tabnet_pca.nlargest(20, "importance"),
                               x="importance", y="feature", orientation="h",
                               title="Top Features - TabNet PCA", template="plotly_white",
                               color_discrete_sequence=[mck_blue])
        st.plotly_chart(fig_fi_tabnet, use_container_width=True)

    with col2:
        st.subheader("LGBM Ensemble")
        fig_fi_lgbm = px.bar(df_fi_lgbm.nlargest(20, "importance"),
                             x="importance", y="feature", orientation="h",
                             title="Top Features - LGBM", template="plotly_white",
                             color_discrete_sequence=["#7C878E"])
        st.plotly_chart(fig_fi_lgbm, use_container_width=True)

    st.info("🧠 Takeaway: Spending limits and PII handling appear as dominant risk contributors across models.")