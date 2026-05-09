import streamlit as st
import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans


st.markdown(
    """
    <style>
    div[data-testid="metric-container"] {
        background-color: #1E1E1E;
        border: 1px solid #333333;
        padding: 15px;
        border-radius: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="NeoBank Dashboard",
    layout="wide"
)

st.title("🏦 NeoBank AI Banking Dashboard")
# -----------------------------------
# SIDEBAR
# -----------------------------------
st.sidebar.title("🏦 NeoBank Menu")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Financial Overview",
        "🤖 Machine Learning",
        "🧠 Customer Segmentation"
    ]
)

# -----------------------------------
# DATABASE CONNECTION
# -----------------------------------
conn = sqlite3.connect("neobank_europe_v3.db")

@st.cache_data
def load_data():

    conn = sqlite3.connect("neobank_europ_v3.db")

    loans = pd.read_sql("SELECT * FROM loans", conn)
    deposits = pd.read_sql("SELECT * FROM deposits", conn)
    customers = pd.read_sql("SELECT * FROM customers", conn)

    return loans, deposits, customers

# -----------------------------------
# LOAD DATA
# -----------------------------------
loans = pd.read_sql("SELECT * FROM loans", conn)
deposits = pd.read_sql("SELECT * FROM deposits", conn)
customers = pd.read_sql("SELECT * FROM customers", conn)

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table';",
    conn
)

st.write(tables)

# تنظيف أسماء الأعمدة
loans.columns = loans.columns.str.strip()
deposits.columns = deposits.columns.str.strip()
customers.columns = customers.columns.str.strip()

# -----------------------------------
# LOAN PROFIT CALCULATION
# -----------------------------------
loans["r"] = loans["interest_rate"] / 100

loans["future_value"] = (
    loans["amount"] * (1 + loans["r"]) ** 5
)

loans["profit"] = (
    loans["future_value"] - loans["amount"]
)

total_loans = loans["amount"].sum()
total_profit = loans["profit"].sum()

# -----------------------------------
# DEPOSIT COST CALCULATION
# -----------------------------------
deposits["r"] = deposits["interest_rate"] / 100

deposits["future_value"] = (
    deposits["amount"] * (1 + deposits["r"]) ** 5
)

deposits["cost"] = (
    deposits["future_value"] - deposits["amount"]
)

total_deposits = deposits["amount"].sum()
total_cost = deposits["cost"].sum()

# -----------------------------------
# NET PROFIT
# -----------------------------------
net_profit = total_profit - total_cost

# -----------------------------------
# KPI SECTION
# -----------------------------------
if page == "📊 Financial Overview":
    st.subheader("📊 Banking KPIs")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Total Loans",
            f"€{total_loans:,.0f}"
        )

    with col2:
        st.metric(
            "🏦 Total Deposits",
            f"€{total_deposits:,.0f}"
        )

    with col3:
        st.metric(
            "📈 Loan Profit (5Y)",
            f"€{total_profit:,.0f}"
        )

    with col4:
        st.metric(
            "💸 Deposit Cost (5Y)",
            f"€{total_cost:,.0f}"
        )

    st.metric(
        "💥 Net Profit",
        f"€{net_profit:,.0f}"
    )

    st.divider()

    # -----------------------------------
    # TOP CUSTOMERS
    # -----------------------------------
    customer_profit = loans.groupby(
        "customer_id"
    )["profit"].sum().reset_index()

    top_customers = customer_profit.sort_values(
        by="profit",
        ascending=False
    ).head(10)

    st.subheader("🏆 Top 10 Customers by Profit")

    st.dataframe(top_customers)

    # -----------------------------------
    # AVG INTEREST RATE
    # -----------------------------------
    avg_interest = loans["interest_rate"].mean()

    st.metric(
        "📊 Avg Loan Interest Rate",
        f"{avg_interest:.2f}%"
    )

    # -----------------------------------
    # LOAN TERM DISTRIBUTION
    # -----------------------------------
    st.subheader("📅 Loan Terms Distribution")

    loan_terms = loans["term_years"].value_counts()

    st.bar_chart(loan_terms)

    # -----------------------------------
    # AMOUNT VS PROFIT
    # -----------------------------------
    st.subheader("💡 Loan Amount vs Profit")

    st.scatter_chart(
        loans[["amount", "profit"]]
    )

    # -----------------------------------
    # PROFIT BY COUNTRY
    # -----------------------------------
    merged = loans.merge(
        customers,
        on="customer_id"
    )

    country_profit = merged.groupby(
        "country"
    )["profit"].sum().sort_values(
        ascending=False
    )

    st.subheader("🌍 Profit by Country")

    st.bar_chart(country_profit,use_container_width= True)

# -----------------------------------
# -----------------------------------
# MACHINE LEARNING
# -----------------------------------
elif page == "🤖 Machine Learning":

    st.subheader("🤖 Loan Default Prediction")

    # إنشاء Target
    loans["default"] = np.where(
        (loans["interest_rate"] > 7) &
        (loans["amount"] > 100000),
        1,
        0
    )

    # Features & Target
    features = loans[
        ["amount", "interest_rate", "term_years"]
    ]

    target = loans["default"]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42
    )

    # Train Model
    @st.cache_resource
    def train_model(features, target):

        model = RandomForestClassifier()

        model.fit(features, target)

        return model


    model = train_model(X_train, y_train)

    # Accuracy
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "🤖 Model Accuracy",
        f"{accuracy:.2f}"
    )

    st.divider()

    # -----------------------------------
    # Interactive Prediction
    # -----------------------------------

    loan_amount = st.number_input(
        "Loan Amount (€)",
        min_value=1000,
        max_value=500000,
        value=150000
    )

    interest_rate = st.slider(
        "Interest Rate (%)",
        1.0,
        15.0,
        8.0
    )

    term_years = st.slider(
        "Loan Term (Years)",
        1,
        10,
        5
    )

    # Prediction Sample
    sample = [[
        loan_amount,
        interest_rate,
        term_years
    ]]

    prediction = model.predict(sample)

    # Result
    if prediction[0] == 1:
        st.error("🚨 High Risk Customer")

    else:
        st.success("✅ Safe Customer")

# -----------------------------------
# CUSTOMER SEGMENTATION
# -----------------------------------
elif page == "🧠 Customer Segmentation":
    st.subheader("🧠 Customer Segmentation")

    customer_data = loans.groupby(
        "customer_id"
    ).agg({
        "amount": "sum",
        "interest_rate": "mean",
        "term_years": "mean"
    }).reset_index()

    # KMeans
    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    customer_data["segment"] = kmeans.fit_predict(
        customer_data[
            ["amount", "interest_rate", "term_years"]
        ]
    )


    # Segment Labels
    def label_segment(row):

        if row["segment"] == 0:
            return "💰 High Value"

        elif row["segment"] == 1:
            return "⚖️ Medium Value"

        else:
            return "📉 Low Value"


    customer_data["segment_label"] = customer_data.apply(
        label_segment,
        axis=1
    )

    # Show Data
    st.dataframe(customer_data)

    # -----------------------------------
    # SEGMENT VISUALIZATION
    # -----------------------------------
    st.subheader("📊 Segmentation Visualization")

    st.scatter_chart(
        customer_data[
            ["amount", "interest_rate"]
        ]
    )

    # -----------------------------------
    # SEGMENT VS PROFIT
    # -----------------------------------
    st.subheader("💰 Profit by Segment")

    customer_profit = loans.groupby(
        "customer_id"
    )["profit"].sum().reset_index()

    segmented = customer_data.merge(
        customer_profit,
        on="customer_id"
    )

    segment_profit = segmented.groupby(
        "segment_label"
    )["profit"].sum().sort_values(
        ascending=False
    )

    st.bar_chart(segment_profit)

    # -----------------------------------
    # AVG PROFIT PER SEGMENT
    # -----------------------------------
    st.subheader("📊 Avg Profit per Segment")

    avg_profit = segmented.groupby(
        "segment_label"
    )["profit"].mean().sort_values(
        ascending=False
    )

    st.bar_chart(avg_profit)

    # -----------------------------------
    # SEGMENT SIZE
    # -----------------------------------
    st.subheader("👥 Number of Customers per Segment")

    segment_size = customer_data[
        "segment_label"
    ].value_counts()

    st.bar_chart(segment_size)
