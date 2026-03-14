import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
st.set_page_config(
    page_title="Automatic Feature Dashboard",
    page_icon="🧠",      # emoji o path a un icono .png
    layout="wide",       # puede ser "centered" o "wide"
    initial_sidebar_state="expanded"
)
st.title("Automatic Feature Processing Dashboard")

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv","xlsx"])

if uploaded_file:

    # Detect file type
    file_type = uploaded_file.name.split('.')[-1].lower()

    if file_type == "csv":
        df = pd.read_csv(uploaded_file)

    elif file_type in ["xls","xlsx"]:
        xls = pd.ExcelFile(uploaded_file)
        sheet = st.selectbox("Select sheet", xls.sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)

    st.subheader("Dataset preview")
    st.dataframe(df.head())

    # TARGET SELECTION
    target = st.selectbox("Select target column", df.columns)

    #Ordinal Scaling
    scale_ordinal = st.checkbox(
        "Scale ordinal features",
        value=False,
        help="Tree-based models (Random Forest, XGBoost) usually do not require scaling for ordinal features."
    )

    if target:

        features = df.drop(columns=[target])
        numeric_cols = features.select_dtypes(include=np.number).columns
        categorical_cols = features.select_dtypes(include="object").columns
        #Create ordinals with order matter
        ordinal_cat_cols = st.multiselect(
        "Select ordinal categorical variables",
        categorical_cols,
        help="Select categorical variables where the order matters (e.g. bad < good < excellent)")
        if ordinal_cat_cols:
            st.write("Ordinal categorical variables:", ordinal_cat_cols)
        nominal_cols = [col for col in categorical_cols if col not in ordinal_cat_cols]

        # Detect ordinal vs continuous
        ordinal_cols = [col for col in numeric_cols if features[col].nunique() <= 10]
        continuous_cols = [col for col in numeric_cols if col not in ordinal_cols]

        # Lists used for pipeline
        standard_cols = []
        robust_cols = []
        minmax_cols = []
        results = []
        one_hot_encoding_col=[]
        ordinal_encoding_col=[]


        for col in numeric_cols:

            data = features[col].dropna()
            sk = skew(data)
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            outliers = ((data < (q1 - 1.5*iqr)) | (data > (q3 + 1.5*iqr))).mean()

            # SCALER DECISION + EXPLANATION
            if col in ordinal_cols :
                if scale_ordinal:
                    scaler = "StandardScaler"
                    reason = "Ordinal feature scaled because toggle is enabled"
                    standard_cols.append(col)
                else:
                    scaler = "No scaling"
                    reason = "Ordinal features often work well without scaling"

            elif outliers > 0.05 or abs(sk) > 1:
                scaler = "RobustScaler"
                reason = "More than 5% outliers detected"
                robust_cols.append(col)


            elif data.min() >= 0 and data.max() <= 1:
                scaler = "MinMaxScaler"
                reason = "Feature bounded between 0 and 1"
                minmax_cols.append(col)

            else:
                scaler = "StandardScaler"
                reason = "Continuous variable without strong outliers"
                standard_cols.append(col)

            results.append({
                "Variable": col,
                "Type": "Ordinal" if col in ordinal_cols else "Continuous",
                "Outliers %": round(outliers*100,2),
                "Suggested scaler": scaler,
                "Reason": reason
            })
        for col in categorical_cols:
            if col in ordinal_cat_cols:
                scaler="No scaling"
                reason="Categorical variable where order matters"
                ordinal_encoding_col.append(col)
                var_type="Categorica ordinal"
            else:
                scaler="No scaling"
                reason="Categorical variable where order doesn't matters"
                one_hot_encoding_col.append(col)
                var_type="Categorica nominal"
            results.append({
                "Variable": col,
                "Type": var_type,
                "Outliers %": "Doesn't apply",
                "Suggested scaler": scaler,
                "Reason": reason
            })

        if st.button('Visualize variable plots'):
            # VISUALIZATION
            for col in features:
                st.subheader(col)

                fig, ax = plt.subplots()
                ax.hist(features[col].dropna(), bins=30)
                ax.set_title("Histogram")
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                ax2.boxplot(features[col].dropna(), vert=False)
                ax2.set_title("Boxplot")
                st.pyplot(fig2)
    if st.button('Get scaler recommendation'):
        st.session_state.show_results = True


    if st.session_state.get("show_results", False):        # Show recommendations table
        st.subheader("Scaler Recommendations")

        result_df = pd.DataFrame(results)
        st.dataframe(result_df)

        # PIPELINE GENERATION
        if st.button("Generate preprocessing pipeline"):

            transformers = []

            if continuous_cols:
                transformers.append(
                    ("continuous", StandardScaler(), continuous_cols)
                )

            if ordinal_cols and scale_ordinal:
                transformers.append(
                    ("ordinal", StandardScaler(), ordinal_cols)
                )

            if robust_cols:
                transformers.append(
                    ("robust", RobustScaler(), robust_cols)
                )

            if minmax_cols:
                transformers.append(
                    ("minmax", MinMaxScaler(), minmax_cols)
                )


            if one_hot_encoding_col:
                transformers.append(
                        ("nominal", OneHotEncoder(), list(categorical_cols))
                    )
            if ordinal_encoding_col:
                transformers.append(
                        ("nominal order", OrdinalEncoder(), list(categorical_cols))
                    )
            # Create code representation
            pipeline_code = f"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder

preprocessor = ColumnTransformer({transformers}, remainder="passthrough")
"""

            st.subheader("Generated preprocessing pipeline")
            st.code(pipeline_code, language="python")
