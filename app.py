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
    c2.metric("Revenue from identified customers", f"{rfm['monetary'].sum():,.0f}" if not rfm.empty else "—")
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

    st.write(
        "Predict the probability that a customer will make at least "
        "one purchase within the next 90 days."
    )

    if model is None:
        st.error("Prediction model could not be loaded.")
        st.stop()

    # One threshold shared by both prediction methods
    threshold = st.slider(
        "Prediction threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05
    )

    # Two prediction options
    upload_tab, manual_tab = st.tabs([
        "Upload Customer File",
        "Enter One Customer Manually"
    ])

    # ------------------------------------------------
    # OPTION 1: BULK PREDICTION THROUGH CSV UPLOAD
    # ------------------------------------------------

    with upload_tab:

        st.subheader("Upload customer data")

        st.write(
            "Upload a CSV file containing one row per customer "
            "and the nine model features."
        )

        uploaded_file = st.file_uploader(
            "Choose a customer CSV file",
            type=["csv"],
            key="customer_upload"
        )

        if uploaded_file is not None:

            try:
                uploaded_data = pd.read_csv(uploaded_file)

                st.write("Uploaded data preview:")
                st.dataframe(
                    uploaded_data.head(),
                    use_container_width=True
                )

                # Check that all model features exist
                missing_columns = [
                    col for col in FEATURES
                    if col not in uploaded_data.columns
                ]

                if missing_columns:
                    st.error(
                        f"Your file is missing these required columns: "
                        f"{missing_columns}"
                    )

                elif uploaded_data.empty:
                    st.error("The uploaded file contains no customer records.")

                else:

                    # Convert model features to numeric values
                    customer_features = uploaded_data[FEATURES].apply(
                        pd.to_numeric,
                        errors="coerce"
                    )

                    # Identify values that could not be converted
                    invalid_values = (
                        uploaded_data[FEATURES].notna()
                        & customer_features.isna()
                    )

                    if invalid_values.any().any():
                        st.error(
                            "Some feature values are not numeric. "
                            "Please correct them and upload the file again."
                        )

                    elif customer_features.isna().all().any():
                        st.error(
                            "At least one required feature column is "
                            "completely empty."
                        )

                    elif (customer_features < 0).any().any():
                        st.error(
                            "Feature values cannot be negative."
                        )

                    elif (customer_features["frequency"] < 1).any():
                        st.error(
                            "Frequency must be at least 1."
                        )

                    elif (customer_features["unique_products"] < 1).any():
                        st.error(
                            "Unique products must be at least 1."
                        )

                    elif (customer_features["unique_countries"] < 1).any():
                        st.error(
                            "Unique countries must be at least 1."
                        )

                    else:

                        if customer_features.isna().any().any():
                            st.warning(
                                "Some values are missing. The model's "
                                "median imputer will handle them, provided "
                                "the saved pipeline includes that imputer."
                            )

                        # Generate purchase probabilities
                        probabilities = model.predict_proba(
                            customer_features[FEATURES]
                        )[:, 1]

                        # Preserve customer identifiers where available
                        if "customer_id" in uploaded_data.columns:
                            results = uploaded_data[
                                ["customer_id"]
                            ].copy()
                        else:
                            results = pd.DataFrame({
                                "customer_id": [
                                    f"Customer {i + 1}"
                                    for i in range(len(uploaded_data))
                                ]
                            })

                        results["purchase_probability"] = probabilities

                        results["prediction"] = [
                            "Likely to Purchase"
                            if probability >= threshold
                            else "Less Likely to Purchase"
                            for probability in probabilities
                        ]

                        # Display summary
                        st.subheader("Prediction Summary")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Customers Analysed",
                            len(results)
                        )

                        col2.metric(
                            "Likely to Purchase",
                            (
                                results["prediction"]
                                == "Likely to Purchase"
                            ).sum()
                        )

                        col3.metric(
                            "Less Likely to Purchase",
                            (
                                results["prediction"]
                                == "Less Likely to Purchase"
                            ).sum()
                        )

                        # Display detailed predictions
                        st.subheader("Customer Prediction Results")

                        st.dataframe(
                            results,
                            use_container_width=True,
                            hide_index=True
                        )

                        # Download predictions
                        csv_data = results.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_data,
                            file_name="customer_purchase_predictions.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Unable to process the uploaded file: {e}")

    # ------------------------------------------------
    # OPTION 2: MANUAL SINGLE-CUSTOMER PREDICTION
    # ------------------------------------------------

    with manual_tab:

        st.subheader("Enter customer information")

        st.write(
            "Use this option when you want to predict the purchase "
            "likelihood of just one customer."
        )

        with st.form("prediction_form"):

            col1, col2, col3 = st.columns(3)

            with col1:
                frequency = st.number_input(
                    "Purchase Frequency",
                    min_value=1,
                    value=5,
                    step=1
                )

                monetary = st.number_input(
                    "Total Monetary Value",
                    min_value=0.01,
                    value=1500.0,
                    step=100.0
                )

                total_units = st.number_input(
                    "Total Units Purchased",
                    min_value=1,
                    value=80,
                    step=1
                )

            with col2:
                average_line_revenue = st.number_input(
                    "Average Line Revenue",
                    min_value=0.0,
                    value=18.75,
                    step=1.0
                )

                unique_products = st.number_input(
                    "Unique Products",
                    min_value=1,
                    value=12,
                    step=1
                )

                unique_countries = st.number_input(
                    "Unique Countries",
                    min_value=1,
                    value=1,
                    step=1
                )

            with col3:
                recency_days = st.number_input(
                    "Recency (Days)",
                    min_value=0,
                    value=45,
                    step=1
                )

                customer_tenure_days = st.number_input(
                    "Customer Tenure (Days)",
                    min_value=0,
                    value=365,
                    step=30
                )

                average_order_value = st.number_input(
                    "Average Order Value",
                    min_value=0.0,
                    value=300.0,
                    step=25.0
                )

            submitted = st.form_submit_button(
                "Predict Customer Purchase"
            )

        if submitted:

            values = [
                frequency,
                monetary,
                total_units,
                average_line_revenue,
                unique_products,
                unique_countries,
                recency_days,
                customer_tenure_days,
                average_order_value
            ]

            row = pd.DataFrame(
                [dict(zip(FEATURES, values))]
            )

            probability = float(
                model.predict_proba(row[FEATURES])[0, 1]
            )

            st.subheader("Prediction Result")

            st.metric(
                "Purchase Probability",
                f"{probability:.1%}"
            )

            st.progress(
                max(0.0, min(1.0, probability))
            )

            if probability >= threshold:
                st.success(
                    "This customer is classified as likely to purchase."
                )
            else:
                st.warning(
                    "This customer is classified as less likely to purchase."
                )

            st.caption(
                "This is a probability estimate, not a guarantee "
                "of future purchasing behaviour."
            )

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
