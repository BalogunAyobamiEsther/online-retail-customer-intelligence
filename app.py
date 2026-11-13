from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "purchase_prediction_model.joblib"
DATA = ROOT / "analytics"
FEATURES = [
    "frequency", "monetary", "total_units", "average_line_revenue",
    "unique_products", "unique_countries", "recency_days",
    "customer_tenure_days", "average_order_value"
]

st.set_page_config(page_title="Online Retail Customer Intelligence", page_icon="🛍️", layout="wide")
st.markdown("""
<style>
.main {background:#f7f8fc}.block-container{padding-top:1.7rem;padding-bottom:3rem}
[data-testid="stSidebar"]{background:#102a43}[data-testid="stSidebar"] *{color:#fff}
.hero{padding:2rem 2.2rem;border-radius:18px;background:linear-gradient(120deg,#102a43,#1f6b70);color:#fff;margin-bottom:1.2rem}
.hero h1{color:#fff;margin-bottom:.4rem}.hero p{color:#e6f1f2;font-size:1.04rem}
.note{color:#52606d;font-size:.9rem}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@st.cache_data
def load_csv(filename):
    path = DATA / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

model = load_model()
monthly = load_csv("monthly_sales_summary.csv")
top_products = load_csv("top_products_by_revenue.csv")
countries = load_csv("country_sales_summary.csv")
rfm = load_csv("customer_rfm_segments.csv")
clusters = load_csv("customer_cluster_profile.csv")
comparison = load_csv("purchase_model_comparison.csv")
priority = load_csv("customer_purchase_priority.csv")

with st.sidebar:
    st.markdown("## 🛍️ Customer Intelligence")
    st.caption("Online Retail Analytics & Prediction")
    page = st.radio("Navigate", ["Home", "Sales Analytics", "Customer Segments", "Prediction", "About", "How to Use"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Decision-support prototype • 90-day purchase prediction")

if page == "Home":
    st.markdown("""<div class="hero"><h1>Online Retail Customer Intelligence</h1>
    <p>Explore retail performance, understand customer segments and estimate the likelihood of a future purchase.</p></div>""", unsafe_allow_html=True)
    st.subheader("Project at a glance")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers analysed", f"{len(rfm):,}" if not rfm.empty else "—")
    c2.metric("Gross revenue", f"{rfm['monetary'].sum():,.0f}" if not rfm.empty else "—")
    c3.metric("Products ranked", f"{len(top_products):,}" if not top_products.empty else "—")
    c4.metric("Prediction horizon", "90 days")
    st.markdown("### Explore the application")
    a,b,c = st.columns(3)
    a.markdown("**Sales Analytics**\n\nReview monthly revenue, leading products and country performance.")
    b.markdown("**Customer Segments**\n\nExplore RFM customer profiles and K-Means cluster summaries.")
    c.markdown("**Prediction**\n\nEstimate the probability of a customer making a purchase in the next 90 days.")
    if model is not None: st.success("Trained prediction model loaded successfully.")
    else: st.error("Prediction model file is missing. Check the models folder.")
    st.caption("Source: the supplied Online Retail II notebook outputs. Currency values retain the dataset's original units.")

elif page == "Sales Analytics":
    st.title("Sales Analytics")
    if monthly.empty:
        st.warning("Monthly sales summary is missing from the analytics folder.")
    else:
        st.subheader("Monthly revenue trend")
        chart = monthly.copy()
        chart["invoice_month"] = pd.to_datetime(chart["invoice_month"], errors="coerce")
        chart = chart.dropna(subset=["invoice_month"]).set_index("invoice_month")
        st.line_chart(chart["revenue"])
        x,y,z = st.columns(3)
        x.metric("Total revenue", f"{monthly['revenue'].sum():,.2f}")
        y.metric("Invoices", f"{monthly['invoices'].sum():,.0f}")
        z.metric("Units sold", f"{monthly['units'].sum():,.0f}")
    l,r = st.columns(2)
    with l:
        st.subheader("Top products by revenue")
        if not top_products.empty:
            st.dataframe(top_products.sort_values("revenue", ascending=False).head(10), use_container_width=True, hide_index=True)
    with r:
        st.subheader("Country performance")
        if not countries.empty:
            st.bar_chart(countries.sort_values("revenue", ascending=False).head(10).set_index("country")["revenue"])
            st.dataframe(countries.sort_values("revenue", ascending=False).head(10), use_container_width=True, hide_index=True)

elif page == "Customer Segments":
    st.title("Customer Segmentation")
    if not clusters.empty:
        st.subheader("K-Means cluster profile")
        st.dataframe(clusters, use_container_width=True, hide_index=True)
        st.bar_chart(clusters.set_index("cluster")["customers"])
    if not rfm.empty:
        st.subheader("RFM customer records")
        segment_options = ["All"] + sorted(rfm["segment"].dropna().astype(str).unique().tolist())
        chosen = st.selectbox("Filter by segment", segment_options)
        view = rfm if chosen == "All" else rfm[rfm["segment"].astype(str) == chosen]
        st.caption(f"Showing {len(view):,} of {len(rfm):,} customers")
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download displayed customer records", view.to_csv(index=False).encode("utf-8"), "customer_rfm_filtered.csv", "text/csv")
    if not priority.empty:
        st.subheader("Purchase-priority list")
        st.write("The notebook's exported customer-level priority scores and purchase estimates.")
        st.dataframe(priority.head(25), use_container_width=True, hide_index=True)
        st.download_button("Download full purchase-priority list", priority.to_csv(index=False).encode("utf-8"), "customer_purchase_priority.csv", "text/csv")

elif page == "Prediction":
    st.title("Customer Purchase Prediction")
    st.write("Enter customer behaviour observed at the prediction snapshot. The model estimates the probability of at least one purchase in the following 90 days.")
    if model is None:
        st.error("The trained model could not be found. Confirm that models/purchase_prediction_model.joblib exists.")
        st.stop()
    with st.form("prediction_form"):
        st.subheader("Customer purchase history")
        c1,c2,c3 = st.columns(3)
        with c1:
            frequency = st.number_input("Number of orders (frequency)", min_value=1, value=5, step=1)
            monetary = st.number_input("Total historical spend", min_value=0.01, value=1500.0, step=100.0)
            total_units = st.number_input("Total units purchased", min_value=1, value=80, step=1)
        with c2:
            average_line_revenue = st.number_input("Average line revenue", min_value=0.0, value=18.75, step=1.0)
            unique_products = st.number_input("Unique products purchased", min_value=1, value=12, step=1)
            unique_countries = st.number_input("Unique countries", min_value=1, value=1, step=1)
        with c3:
            recency_days = st.number_input("Days since last purchase", min_value=0, value=45, step=1)
            customer_tenure_days = st.number_input("Customer tenure (days)", min_value=0, value=365, step=30)
            average_order_value = st.number_input("Average order value", min_value=0.0, value=300.0, step=25.0)
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, help="Exploratory threshold only; it has not been operationally validated.")
        submitted = st.form_submit_button("Predict purchase likelihood", type="primary", use_container_width=True)
    if submitted:
        row = pd.DataFrame([dict(zip(FEATURES, [frequency, monetary, total_units, average_line_revenue, unique_products, unique_countries, recency_days, customer_tenure_days, average_order_value]))], columns=FEATURES)
        probability = float(model.predict_proba(row)[0,1])
        left,right = st.columns([1,2])
        with left:
            st.metric("Purchase probability (next 90 days)", f"{probability:.1%}")
            st.progress(max(0.0, min(1.0, probability)))
        with right:
            if probability >= threshold:
                st.success("Higher estimated likelihood of a purchase")
                st.write("The estimated probability is at or above the selected threshold.")
            else:
                st.warning("Lower estimated likelihood of a purchase")
                st.write("The estimated probability is below the selected threshold.")
        st.caption("This is a model estimate, not a guarantee. A low score does not prove that a customer has churned.")

elif page == "About":
    st.title("About the Project")
    st.write("This application packages the supplied Online Retail II notebook outputs into an interactive customer-intelligence dashboard.")
    st.markdown("""
    **Project components**
    - Retail sales analysis: revenue, invoices, products and country summaries.
    - Customer-level analysis: recency, frequency and monetary (RFM) features.
    - Customer segmentation: K-Means clustering on transformed and standardized RFM variables.
    - Purchase prediction: classification of whether a customer makes at least one purchase in a defined 90-day target window.
    """)
    st.subheader("Model comparison from the notebook")
    if not comparison.empty:
        st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.warning("The notebook's held-out evaluation indicates limitations in identifying non-purchasers at the default threshold. Threshold tuning and rolling time-based validation are needed before operational use.")
    st.write("The target is purchasing during one specific 90-day period; it is not a permanent customer churn label. Use scores to support review and controlled campaign tests.")

elif page == "How to Use":
    st.title("How to Use the App")
    st.markdown("""
    1. Install the dependencies using `pip install -r requirements.txt`.
    2. Start the app using `streamlit run app.py`.
    3. Use the sidebar to open **Sales Analytics**, **Customer Segments** or **Prediction**.
    4. For prediction, enter all nine customer metrics using the same definitions and units as the notebook.
    5. Adjust the decision threshold only to explore how the displayed class changes.
    """)
    st.info("The trained model and exported analytics are already included. You do not need to rerun the notebook or train the model to launch this packaged app.")
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")
    st.caption("For deployment, keep the app.py, requirements.txt, models/ and analytics/ paths together in the repository.")
