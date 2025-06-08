import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("processed_delivery_data-cp.csv")

df = load_data()

# Streamlit UI
st.title("📊 Amazon Delivery Dataset - EDA Dashboard")
st.markdown("This dashboard provides an overview and analysis of delivery-related features.")

# Sidebar options
st.sidebar.header("EDA Options")
show_data = st.sidebar.checkbox("Show Raw Data")
show_stats = st.sidebar.checkbox("Show Summary Statistics")
show_corr = st.sidebar.checkbox("Show Correlation Heatmap")
show_dist = st.sidebar.checkbox("Show Distributions (Numerical)")
show_cat = st.sidebar.checkbox("Show Count Plots (Categorical)")
show_box = st.sidebar.checkbox("Show Boxplots for Outliers")
show_pair = st.sidebar.checkbox("Show Pairplot (slow)", False)

# Show raw data
if show_data:
    st.subheader("📄 Raw Dataset")
    st.dataframe(df)

# Summary statistics
if show_stats:
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

# Correlation heatmap
if show_corr:
    st.subheader("📉 Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(plt.gcf())
    plt.clf()

# Distribution plots for numerical columns
if show_dist:
    st.subheader("📌 Distribution Plots")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
        st.pyplot(fig)

# Count plots for categorical columns
if show_cat:
    st.subheader("🔢 Count Plots (Categorical Features)")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f'Count Plot for {col}')
        plt.xticks(rotation=45)
        st.pyplot(fig)

# Boxplots to detect outliers
if show_box:
    st.subheader("📦 Boxplots for Outlier Detection")
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f'Boxplot for {col}')
        st.pyplot(fig)

# Pairplot (optional, slow)
if show_pair:
    st.subheader("🧩 Pairplot (All Numerical Features)")
    st.info("This may take a few seconds for large datasets.")
    fig = sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
    st.pyplot(fig)
