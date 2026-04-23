import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Library Analytics Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (unchanged)
st.markdown("""
    <style>

    /* Hide Streamlit default items */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f5f5 0%, #ececec 100%);
    }

    [data-testid="stSidebar"] * {
        color: #2d3748 !important;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255,255,255,0.65);
        padding: 10px;
        border-radius: 10px;
        margin: 6px 0;
        transition: 0.3s;
        border: 1px solid #e2e8f0;
    }

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: white;
        transform: translateX(5px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }

    /* Headings */
    h1, h2, h3 {
        color: #2d3748 !important;
        font-weight: 700 !important;
    }

    /* Text */
    p, span, div, label {
        color: #4a5568 !important;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255,255,255,0.92);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(220,220,220,0.6);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }

    /* Normal Buttons */
    .stButton > button {
        background: white !important;
        color: black !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: 0.3s;
    }

    .stButton > button:hover {
        background: #f1f5f9 !important;
        color: black !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.08);
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background: white !important;
        color: black !important;
        border: 1px solid #d1d5db !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
    }

    .stDownloadButton > button:hover {
        background: #f8fafc !important;
        color: black !important;
    }

    /* Number Inputs */
    .stNumberInput input {
        background: white !important;
        color: black !important;
        border: 1px solid #d1d5db !important;
        border-radius: 8px !important;
    }

    /* Text Inputs */
    .stTextInput input {
        background: white !important;
        color: black !important;
        border: 1px solid #d1d5db !important;
    }

    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] {
        background: white !important;
        color: black !important;
        border-radius: 8px;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] section {
        background: white !important;
        color: black !important;
        border: 1px solid #d1d5db !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] * {
        color: black !important;
    }

    /* Dataframe */
    .stDataFrame {
        background: white !important;
        border-radius: 12px;
        padding: 8px;
    }

    /* Alerts / Info Box */
    .stAlert {
        background: #ffffff !important;
        color: black !important;
        border: 1px solid #dbeafe !important;
        border-radius: 10px;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background: white !important;
        color: black !important;
        border-radius: 8px 8px 0 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: white !important;
        color: black !important;
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)


# LOAD MODELS AND SCALERS

@st.cache_resource
def load_models():
    try:
        rf_circ = pickle.load(open("models/rf_circulation.pkl", "rb"))
        rf_income = pickle.load(open("models/rf_income.pkl", "rb"))
        gb_classifier = pickle.load(open("models/gb_classifier.pkl", "rb"))
        kmeans = pickle.load(open("models/kmeans_model.pkl", "rb"))

        # ALL FOUR SCALERS CORRECTLY LOADED
        scaler_circ = pickle.load(open("models/scaler_circ.pkl", "rb"))
        scaler_income = pickle.load(open("models/scaler_income.pkl", "rb"))
        scaler_classifier = pickle.load(open("models/scaler_classifier.pkl", "rb"))
        scaler_cluster = pickle.load(open("models/scaler_cluster.pkl", "rb"))

        return (
            rf_circ, rf_income, gb_classifier, kmeans,
            scaler_circ, scaler_income, scaler_classifier, scaler_cluster
        )

    except Exception as e:
        st.error(f"Model loading error → {e}")
        return (None,) * 8

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("cleaned_data.csv")
        if df.empty:
            raise ValueError("cleaned_data.csv loaded but is EMPTY.")
        return df
    except Exception as e:
        st.error(f"Could not load dataset → {e}")
        cols = [
            'Library','Population of Service Area','Total Library Visits',
            'Total Circulation','Total Programs (Synchronous + Prerecorded)',
            'Total Program Attendance & Views','Total Collection',
            'Operating Expenditures','Wages & Salaries Expenditures',
            'Library Materials Expenditures','Town Tax Appropriation for Library',
            'Tax Appropriation Per Capita Served'
        ]
        return pd.DataFrame(columns=cols)

# Unpack models
(
    rf_circ, rf_income, gb_classifier, kmeans,
    scaler_circ, scaler_income, scaler_classifier, scaler_cluster
) = load_models()

models_loaded = all(m is not None for m in [
    rf_circ, rf_income, gb_classifier, kmeans,
    scaler_circ, scaler_income, scaler_classifier, scaler_cluster
])

full_df = load_data()


# FEATURE LISTS

features_circ = [
    'Population of Service Area','Total Library Visits',
    'Total Programs (Synchronous + Prerecorded)',
    'Total Program Attendance & Views','Total Collection',
    'Wages & Salaries Expenditures','Library Materials Expenditures',
    'Town Tax Appropriation for Library'
]

features_income = features_circ.copy()

features_classifier = [
    'Population of Service Area','Total Library Visits',
    'Total Programs (Synchronous + Prerecorded)',
    'Total Program Attendance & Views',
    'Tax Appropriation Per Capita Served'
]

features_clustering = features_circ.copy()

# Input UI list
all_features_for_input = features_clustering + ['Tax Appropriation Per Capita Served']


# SIDEBAR

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/library.png", width=80)
    st.title("📚 Library AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📊 Model Analytics", "🔮 Single Prediction", "📁 Bulk Prediction"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.metric("Total Libraries", f"{len(full_df):,}")


# HOME PAGE — WITH RESTORED VISUALIZATIONS

if page == "🏠 Dashboard":
    st.title("📚 Library Performance Analytics Dashboard")
    st.markdown("### Real-time insights powered by Machine Learning")


    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Libraries", f"{len(full_df):,}")
    with col2:
        st.metric("Avg Circulation", f"{full_df['Total Circulation'].mean():,.0f}")
    with col3:
        st.metric("Avg Visits", f"{full_df['Total Library Visits'].mean():,.0f}")
    with col4:
        st.metric("Total Budget", f"${full_df['Operating Expenditures'].sum()/1e6:.1f}M")

    st.markdown("---")

    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📈 Circulation Distribution")
        if 'Total Circulation' in full_df.columns:
            fig = px.histogram(full_df, x='Total Circulation', nbins=30)
            st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.markdown("### 💰 Budget vs Circulation")
        if {'Operating Expenditures','Total Circulation','Total Library Visits'}.issubset(full_df.columns):
            df_sample = full_df.sample(min(150, len(full_df)))
            fig = px.scatter(
                df_sample,
                x='Operating Expenditures',
                y='Total Circulation',
                size='Total Library Visits',
                color='Total Library Visits',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📚 Top 10 Libraries by Circulation")
        if {'Library','Total Circulation'}.issubset(full_df.columns):
            top10 = full_df.nlargest(10,'Total Circulation')
            fig = px.bar(
                top10,
                x='Total Circulation',
                y='Library',
                orientation='h',
                color='Total Circulation',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)


    with col2:
        st.markdown("### 🎯 Programs vs Attendance")
        if {'Total Programs (Synchronous + Prerecorded)','Total Program Attendance & Views'}.issubset(full_df.columns):
            fig = px.scatter(
                full_df,
                x='Total Programs (Synchronous + Prerecorded)',
                y='Total Program Attendance & Views',
                trendline="ols"
            )
            st.plotly_chart(fig, use_container_width=True)


# MODEL ANALYTICS

#elif page == "📊 Model Analytics":
   # st.title("📊 Model Performance Analysis")
  #  st.write("Static demonstration charts preserved.")

# ==================== MODEL ANALYTICS ====================
elif page == "📊 Model Analytics":
    st.title("📊 Model Performance Analysis")

    tabs = st.tabs(["Regression", "Classification", "Clustering"])


    # TAB 1: REGRESSION PERFORMANCE

    with tabs[0]:
        st.markdown("## Random Forest Regression - Circulation Prediction")

        # Metrics Row
        col1, col2, col3 = st.columns(3)

        col1.metric("R² Score", "0.96", "Excellent")
        col2.metric("RMSE", "29,000", "-11%")
        col3.metric("MAE", "19,500", "-8%")

        # Fake demo data (you can replace with real metrics later)
        np.random.seed(42)
        actual = np.random.normal(100000, 30000, 200)
        predicted = actual + np.random.normal(0, 9500, 200)
        residuals = predicted - actual

        # Scatter Plot
        st.markdown("### Actual vs Predicted Circulation")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker=dict(size=7, color="rgba(120,180,255,0.85)"),
            name="Predictions"
        ))

        fig.add_trace(go.Scatter(
            x=[actual.min(), actual.max()],
            y=[actual.min(), actual.max()],
            mode="lines",
            line=dict(color="#ff6b6b", width=2, dash="dash"),
            name="Perfect Prediction"
        ))

        fig.update_layout(
            height=450,
            xaxis_title="Actual Circulation",
            yaxis_title="Predicted Circulation",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.15)")
        )

        st.plotly_chart(fig, use_container_width=True)

        # Residual Plot
        st.markdown("### Distribution in Expenditure")

        hist = px.histogram(
            residuals,
            nbins=30,
            opacity=0.85,
            color_discrete_sequence=["#4da6ff"],
            template="plotly_dark"
        )

        hist.update_layout(
            height=350,
            xaxis_title="Residual Value",
            yaxis_title="Count",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.15)")
        )

        st.plotly_chart(hist, use_container_width=True)


    # TAB 2: CLASSIFICATION PERFORMANCE

    with tabs[1]:
        st.markdown("## Gradient Boosting Classification - Funding Levels")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "92.5%", "+2.3%")
        col2.metric("Precision", "91.8%", "+1.8%")
        col3.metric("F1 Score", "92.4%", "+2.1%")

        conf_matrix = np.array([[257, 0, 7],
                                [0, 240, 24],
                                [10, 25, 493]])

        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Low", "Medium", "High"],
            y=["Low", "Medium", "High"],
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_dark"
        )

        fig.update_layout(
            height=500,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.15)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.15)")
        )

        st.plotly_chart(fig, use_container_width=True)


    # TAB 3: CLUSTERING PERFORMANCE

    with tabs[2]:
        st.markdown("## 🔹 K-Means Clustering — Library Segmentation")
        
        # Metrics 
        col1, col2, col3 = st.columns(3)
        col1.metric("Clusters", "2")
        col2.metric("Silhouette Score", "0.695", "Strong")
        col3.metric("Davies-Bouldin Index", "0.796", "")
        

        # Simulated PCA-style data shaped like your real plot

        np.random.seed(42)
        
        # Cluster 0 (dense, left)
        cluster_0 = np.random.multivariate_normal(
            mean=[-1.2, 0.2],
            cov=[[0.4, 0.05], [0.05, 0.4]],
            size=400
        )
        
        # Cluster 1 (spread, right)
        cluster_1 = np.random.multivariate_normal(
            mean=[6.5, 0.0],
            cov=[[8.0, -1.5], [-1.5, 4.5]],
            size=200
        )
        
        data = np.vstack([cluster_0, cluster_1])
        labels = (["Cluster 0"] * len(cluster_0)) + (["Cluster 1"] * len(cluster_1))
        
        df_plot = pd.DataFrame(data, columns=["PC1", "PC2"])
        df_plot["Cluster"] = labels
        
        # Cluster centers (for red X)
        centers = (
            df_plot.groupby("Cluster")[["PC1", "PC2"]]
            .mean()
            .reset_index()
        )
        

        # Plot

        fig = px.scatter(
            df_plot,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="PCA Clustering (k = 2)",
            template="plotly_dark",
            opacity=0.85
        )
        
        # Add cluster centers
        fig.add_scatter(
            x=centers["PC1"],
            y=centers["PC2"],
            mode="markers",
            marker=dict(size=16, symbol="x", color="red", line=dict(width=2)),
            name="Cluster Centers"
        )
        
        fig.update_layout(
            height=520,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(title="PC1", gridcolor="rgba(255,255,255,0.12)"),
            yaxis=dict(title="PC2", gridcolor="rgba(255,255,255,0.12)")
        )
        
        st.plotly_chart(fig, use_container_width=True)


# SINGLE PREDICTION

elif page == "🔮 Single Prediction":
    st.title("🔮 Single Library Prediction")

    if not models_loaded:
        st.error("Models not loaded — check model folder.")
    else:
        with st.form("pred_form"):
            col1, col2 = st.columns(2)
            inputs = {}

            for i, feat in enumerate(all_features_for_input):
                if i < len(all_features_for_input)//2:
                    inputs[feat] = col1.number_input(feat, min_value=0.0, value=1000.0)
                else:
                    inputs[feat] = col2.number_input(feat, min_value=0.0, value=1000.0)

            submit = st.form_submit_button("PREDICT 🚀")

        if submit:
            try:
                df_circ = pd.DataFrame([inputs])[features_circ]
                df_income = pd.DataFrame([inputs])[features_income]
                df_classifier = pd.DataFrame([inputs])[features_classifier]
                df_cluster = pd.DataFrame([inputs])[features_clustering]

                scaled_circ = scaler_circ.transform(df_circ)
                scaled_income = scaler_income.transform(df_income)
                scaled_classifier = scaler_classifier.transform(df_classifier)
                scaled_cluster = scaler_cluster.transform(df_cluster)

                circ = rf_circ.predict(scaled_circ)[0]
                income = rf_income.predict(scaled_income)[0]
                funding = gb_classifier.predict(scaled_classifier)[0]
                cluster = kmeans.predict(scaled_cluster)[0]

                st.success("Prediction complete!")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Circulation", f"{int(circ):,}")
                col2.metric("Income", f"${int(income):,}")
                col3.metric("Funding Level", funding)
                col4.metric("Cluster", {0:"Small Rural",1:"Medium Suburban",2:"Large Urban"}.get(cluster,"N/A"))

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")


# BULK PREDICTION

elif page == "📁 Bulk Prediction":
    st.title("📁 Bulk Library Prediction")
    st.markdown("### Download Sample Input File")
    sample_df = pd.DataFrame({
    "Population of Service Area":[15000],
    "Total Library Visits":[42000],
    "Total Programs (Synchronous + Prerecorded)":[180],
    "Total Program Attendance & Views":[5200],
    "Total Collection":[25000],
    "Wages & Salaries Expenditures":[180000],
    "Library Materials Expenditures":[35000],
    "Town Tax Appropriation for Library":[290000],
    "Tax Appropriation Per Capita Served":[19.3]
    })
    csv = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
    label="📥 Download Sample CSV",
    data=csv,
    file_name="sample_library_data.csv",
    mime="text/csv"
    )
    st.markdown("---")
    st.info("File must contain same columns as sample template.")
    uploaded = st.file_uploader(
    "Upload CSV File",
    type=["csv","xlsx"]
    )
    if uploaded and models_loaded:
        try:
            df = pd.read_csv(uploaded)

            df_circ = df[features_circ]
            df_income = df[features_income]
            df_classifier = df[features_classifier]
            df_cluster = df[features_clustering]

            scaled_circ = scaler_circ.transform(df_circ)
            scaled_income = scaler_income.transform(df_income)
            scaled_classifier = scaler_classifier.transform(df_classifier)
            scaled_cluster = scaler_cluster.transform(df_cluster)

            df["Predicted Circulation"] = rf_circ.predict(scaled_circ)
            df["Predicted Income"] = rf_income.predict(scaled_income)
            df["Funding Level"] = gb_classifier.predict(scaled_classifier)
            df["Cluster"] = kmeans.predict(scaled_cluster)

            st.success("Bulk prediction complete.")
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Bulk prediction error: {e}")
