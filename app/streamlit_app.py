import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    preprocessor = joblib.load(
        os.path.join(MODELS_DIR, "preprocessor.pkl")
    )

    classification_models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODELS_DIR, "logistic_regression.pkl")
        ),
        "Decision Tree": joblib.load(
            os.path.join(MODELS_DIR, "decision_tree.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODELS_DIR, "random_forest.pkl")
        ),
        "Extra Trees": joblib.load(
            os.path.join(MODELS_DIR, "extra_trees.pkl")
        ),
        "XGBoost": joblib.load(
            os.path.join(MODELS_DIR, "xgboost_classifier.pkl")
        )
    }

    regression_models = {
        "Linear Regression": joblib.load(
            os.path.join(MODELS_DIR, "linear_regression.pkl")
        ),
        "Decision Tree": joblib.load(
            os.path.join(MODELS_DIR, "decision_tree_regressor.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODELS_DIR, "random_forest_regressor.pkl")
        ),
        "Extra Trees": joblib.load(
            os.path.join(MODELS_DIR, "extra_trees_regressor.pkl")
        ),
        "XGBoost": joblib.load(
            os.path.join(MODELS_DIR, "xgboost_regressor.pkl")
        )
    }

    return preprocessor, classification_models, regression_models


# ============================================================
# LOAD MODELS SAFELY
# ============================================================

try:

    preprocessor, classification_models, regression_models = load_models()

except Exception as e:

    st.error("Unable to load the trained models.")

    st.exception(e)

    st.stop()


# ============================================================
# TITLE
# ============================================================

st.title("🏠 Real Estate Investment Advisor")

st.markdown(
    """
    ### Predict Property Investment Potential & Future Value

    Enter the details of a property to estimate:

    - 📊 Investment potential
    - 🎯 Classification confidence
    - 💰 Estimated 5-year property value
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Model Settings")

classification_model_name = st.sidebar.selectbox(
    "Classification Model",
    list(classification_models.keys()),
    index=4
)

regression_model_name = st.sidebar.selectbox(
    "Regression Model",
    list(regression_models.keys()),
    index=4
)

st.sidebar.divider()

st.sidebar.info(
    """
    The application uses the machine-learning models
    trained during the project.

    Classification target:
    Good_Investment

    Regression target:
    Future_Price_5Y
    """
)


# ============================================================
# PROPERTY DETAILS
# ============================================================

st.header("🏠 Property Details")

col1, col2, col3 = st.columns(3)

with col1:

    state = st.text_input(
        "State",
        value="Bihar"
    )

    city = st.text_input(
        "City",
        value="Katihar"
    )

    locality = st.text_input(
        "Locality",
        value="Main City"
    )

    property_type = st.selectbox(
        "Property Type",
        [
            "Apartment",
            "Independent House",
            "Villa",
            "Builder Floor",
            "Penthouse",
            "Studio"
        ]
    )


with col2:

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

    size_sqft = st.number_input(
        "Size (SqFt)",
        min_value=100.0,
        max_value=10000.0,
        value=1000.0,
        step=50.0
    )

    price_lakhs = st.number_input(
        "Price (₹ Lakhs)",
        min_value=1.0,
        max_value=1000.0,
        value=50.0,
        step=1.0
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1900,
        max_value=2025,
        value=2018,
        step=1
    )


with col3:

    furnished_status = st.selectbox(
        "Furnished Status",
        [
            "Unfurnished",
            "Semi-Furnished",
            "Furnished"
        ]
    )

    floor_no = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=2,
        step=1
    )

    total_floors = st.number_input(
        "Total Floors",
        min_value=1,
        max_value=100,
        value=5,
        step=1
    )

    facing = st.selectbox(
        "Facing",
        [
            "North",
            "South",
            "East",
            "West",
            "North-East",
            "North-West",
            "South-East",
            "South-West"
        ]
    )


# ============================================================
# LOCATION & AMENITIES
# ============================================================

st.divider()

st.header("📍 Location & Amenities")

col1, col2, col3 = st.columns(3)

with col1:

    nearby_schools = st.number_input(
        "Nearby Schools",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    nearby_hospitals = st.number_input(
        "Nearby Hospitals",
        min_value=0,
        max_value=50,
        value=2,
        step=1
    )

    public_transport = st.selectbox(
        "Public Transport Accessibility",
        [
            "Low",
            "Medium",
            "High"
        ]
    )


with col2:

    parking_space = st.selectbox(
        "Parking Space",
        [
            "Yes",
            "No"
        ]
    )

    security = st.selectbox(
        "Security",
        [
            "Yes",
            "No"
        ]
    )

    owner_type = st.selectbox(
        "Owner Type",
        [
            "First Owner",
            "Second Owner",
            "Third Owner"
        ]
    )


with col3:

    availability_status = st.selectbox(
        "Availability Status",
        [
            "Ready to Move",
            "Under Construction"
        ]
    )

    amenities = st.multiselect(
        "Amenities",
        [
            "Swimming Pool",
            "Gym",
            "Garden",
            "Club House",
            "Lift",
            "Power Backup",
            "Security",
            "CCTV",
            "Parking",
            "Playground",
            "Community Hall",
            "WiFi"
        ]
    )


st.divider()


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze = st.button(
    "🔍 Analyze Property",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze:

    try:

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # ----------------------------------------------------

        price_per_sqft = (
            price_lakhs * 100000
        ) / size_sqft

        amenity_density_score = len(amenities)

        price_per_bhk = (
            price_lakhs / bhk
        )

        size_per_bhk = (
            size_sqft / bhk
        )

        property_age = (
            2025 - year_built
        )

        if property_age <= 5:
            property_age_category = "New"

        elif property_age <= 10:
            property_age_category = "Recent"

        elif property_age <= 20:
            property_age_category = "Moderate"

        elif property_age <= 30:
            property_age_category = "Old"

        else:
            property_age_category = "Very_Old"

        if floor_no <= total_floors:
            floor_consistency = "Consistent"
        else:
            floor_consistency = "Inconsistent"


        # ----------------------------------------------------
        # AMENITIES STRING
        # ----------------------------------------------------

        amenities_string = ", ".join(amenities)


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            {
                "State": [state],
                "City": [city],
                "Locality": [locality],
                "Property_Type": [property_type],
                "BHK": [bhk],
                "Size_in_SqFt": [size_sqft],
                "Price_in_Lakhs": [price_lakhs],
                "Price_per_SqFt": [price_per_sqft],
                "Year_Built": [year_built],
                "Furnished_Status": [furnished_status],
                "Floor_No": [floor_no],
                "Total_Floors": [total_floors],
                "Age_of_Property": [property_age],
                "Nearby_Schools": [nearby_schools],
                "Nearby_Hospitals": [nearby_hospitals],
                "Public_Transport_Accessibility": [
                    public_transport
                ],
                "Parking_Space": [parking_space],
                "Security": [security],
                "Facing": [facing],
                "Owner_Type": [owner_type],
                "Availability_Status": [
                    availability_status
                ],
                "Price_per_BHK": [price_per_bhk],
                "Size_per_BHK": [size_per_bhk],
                "Property_Age_Category": [
                    property_age_category
                ],
                "Floor_Consistency": [
                    floor_consistency
                ],
                "Amenity_Density_Score": [
                    amenity_density_score
                ],
                "Amenities": [amenities_string]
            }
        )


        # ----------------------------------------------------
        # REMOVE RAW AMENITIES
        # ----------------------------------------------------

        # The preprocessing notebook excluded the raw
        # Amenities column because it is a multi-value field.

        if "Amenities" in input_data.columns:
            input_data = input_data.drop(
                columns=["Amenities"]
            )


        # ----------------------------------------------------
        # MAKE SURE COLUMN ORDER MATCHES TRAINING DATA
        # ----------------------------------------------------

        expected_columns = [
            "State",
            "City",
            "Locality",
            "Property_Type",
            "BHK",
            "Size_in_SqFt",
            "Price_in_Lakhs",
            "Price_per_SqFt",
            "Year_Built",
            "Furnished_Status",
            "Floor_No",
            "Total_Floors",
            "Age_of_Property",
            "Nearby_Schools",
            "Nearby_Hospitals",
            "Public_Transport_Accessibility",
            "Parking_Space",
            "Security",
            "Facing",
            "Owner_Type",
            "Availability_Status",
            "Price_per_BHK",
            "Size_per_BHK",
            "Property_Age_Category",
            "Floor_Consistency",
            "Amenity_Density_Score"
        ]

        input_data = input_data[
            expected_columns
        ]


        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        processed_input = preprocessor.transform(
            input_data
        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        classifier = classification_models[
            classification_model_name
        ]

        investment_prediction = classifier.predict(
            processed_input
        )[0]


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if hasattr(
            classifier,
            "predict_proba"
        ):

            probabilities = classifier.predict_proba(
                processed_input
            )[0]

            confidence = (
                max(probabilities) * 100
            )

        else:

            confidence = None


        # ====================================================
        # REGRESSION
        # ====================================================

        regressor = regression_models[
            regression_model_name
        ]

        future_price = regressor.predict(
            processed_input
        )[0]

        future_price = max(
            0,
            future_price
        )


        # ====================================================
        # RESULTS
        # ====================================================

        st.divider()

        st.header("📊 Investment Analysis")


        # ----------------------------------------------------
        # INVESTMENT RESULT
        # ----------------------------------------------------

        if investment_prediction == 1:

            st.success(
                "🟢 GOOD INVESTMENT"
            )

            investment_text = (
                "The selected classification model "
                "classified this property as a "
                "Good Investment."
            )

        else:

            st.warning(
                "🟠 LOWER INVESTMENT POTENTIAL"
            )

            investment_text = (
                "The selected classification model "
                "did not classify this property as "
                "a Good Investment."
            )

        st.write(investment_text)


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Current Price",
                f"₹{price_lakhs:.2f} L"
            )

        with col2:

            st.metric(
                "Price / SqFt",
                f"₹{price_per_sqft:,.0f}"
            )

        with col3:

            st.metric(
                "5-Year Estimated Value",
                f"₹{future_price:.2f} L"
            )

        with col4:

            if confidence is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            else:

                st.metric(
                    "Prediction Confidence",
                    "N/A"
                )


        # ====================================================
        # PRICE GROWTH
        # ====================================================

        growth_amount = (
            future_price - price_lakhs
        )

        growth_percentage = (
            growth_amount / price_lakhs
        ) * 100


        st.subheader(
            "💰 5-Year Value Projection"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Estimated Value Increase",
                f"₹{growth_amount:.2f} L"
            )

        with col2:

            st.metric(
                "Estimated Growth",
                f"{growth_percentage:.2f}%"
            )


        # ====================================================
        # PROPERTY SUMMARY
        # ====================================================

        st.subheader(
            "🏠 Property Summary"
        )

        summary_data = pd.DataFrame(
            {
                "Feature": [
                    "State",
                    "City",
                    "Locality",
                    "Property Type",
                    "BHK",
                    "Size",
                    "Current Price",
                    "Property Age",
                    "Amenity Count",
                    "Nearby Schools",
                    "Nearby Hospitals",
                    "Transport Accessibility"
                ],

                "Value": [
                    state,
                    city,
                    locality,
                    property_type,
                    bhk,
                    f"{size_sqft:,.0f} SqFt",
                    f"₹{price_lakhs:.2f} L",
                    f"{property_age} years",
                    amenity_density_score,
                    nearby_schools,
                    nearby_hospitals,
                    public_transport
                ]
            }
        )

        st.dataframe(
            summary_data,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.subheader(
            "🤖 Models Used"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                f"""
                **Investment Classification**

                Model: **{classification_model_name}**

                Target: `Good_Investment`
                """
            )

        with col2:

            st.info(
                f"""
                **Future Price Prediction**

                Model: **{regression_model_name}**

                Target: `Future_Price_5Y`
                """
            )


        # ====================================================
        # ENGINEERED FEATURES
        # ====================================================

        with st.expander(
            "🔧 View Engineered Features"
        ):

            engineered_data = pd.DataFrame(
                {
                    "Feature": [
                        "Price per SqFt",
                        "Price per BHK",
                        "Size per BHK",
                        "Amenity Density Score",
                        "Property Age Category",
                        "Floor Consistency"
                    ],

                    "Value": [
                        f"₹{price_per_sqft:,.2f}",
                        f"₹{price_per_bhk:.2f} L",
                        f"{size_per_bhk:.2f} SqFt",
                        amenity_density_score,
                        property_age_category,
                        floor_consistency
                    ]
                }
            )

            st.dataframe(
                engineered_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)