
import streamlit as st
import pandas as pd
import joblib

from notebooks.feature_engineering import create_features


# ============================================================
# LOAD MODEL AND DATA
# ============================================================

model = joblib.load("models/house_price_model.joblib")

data = pd.read_csv("data/House_Prices.csv")


# Original features used as user inputs
feature_columns = [
    column
    for column in data.columns
    if column not in ["Id", "SalePrice"]
]


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price Prediction")

st.write(
    "Enter the characteristics of the house "
    "to estimate its sale price."
)


# ============================================================
# FORM
# ============================================================

with st.form("house_form"):

    st.subheader("House characteristics")

    # Store user inputs here
    house_values = {}

    # Create 3 columns to make the form smaller
    columns = st.columns(3)

    for index, feature in enumerate(feature_columns):

        # Put the widget in one of the 3 columns
        column = columns[index % 3]

        with column:

            # ------------------------------------------------
            # NUMERICAL FEATURE
            # ------------------------------------------------

            if pd.api.types.is_numeric_dtype(data[feature]):

                # Use median as the default value
                default_value = data[feature].median()

                # Integer column
                if pd.api.types.is_integer_dtype(data[feature]):

                    house_values[feature] = st.number_input(
                        feature,
                        min_value=int(data[feature].min()),
                        max_value=int(data[feature].max()),
                        value=int(default_value),
                        step=1
                    )

                # Float column
                else:

                    house_values[feature] = st.number_input(
                        feature,
                        min_value=float(data[feature].min()),
                        max_value=float(data[feature].max()),
                        value=float(default_value),
                        step=0.1
                    )

            # ------------------------------------------------
            # CATEGORICAL FEATURE
            # ------------------------------------------------

            else:

                # Get existing categories from the dataset
                options = (
                    data[feature]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                options.sort()

                # Most frequent category as default
                mode = data[feature].mode()

                if len(mode) > 0:
                    default_value = str(mode.iloc[0])
                else:
                    default_value = options[0]

                default_index = options.index(default_value)

                house_values[feature] = st.selectbox(
                    feature,
                    options=options,
                    index=default_index
                )

    # ========================================================
    # SUBMIT BUTTON
    # ========================================================

    submitted = st.form_submit_button(
        "Predict House Price",
        type="primary"
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    # Convert the dictionary into a DataFrame
    house = pd.DataFrame(
        [house_values],
        columns=feature_columns
    )

    # Make sure numeric columns have the same types
    for column in feature_columns:

        if pd.api.types.is_numeric_dtype(data[column]):
            house[column] = pd.to_numeric(house[column])

    # --------------------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------------------

    house = create_features(house)

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(house)

    predicted_price = prediction[0]

    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.success(
        f"Estimated Sale Price: ${predicted_price:,.2f}"
    )

    # Optional: show what was sent to the model
    with st.expander("Show input data"):
        st.dataframe(house)

