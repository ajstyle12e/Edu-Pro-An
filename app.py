import streamlit as st
import pandas as pd
import numpy as np
import joblib

import plotly.express as px

st.set_page_config(
    page_title="EduPro Predictive Analytics",
    page_icon="📚",
    layout="wide"
)

@st.cache_resource
def load_models():

    enrollment_model = joblib.load(
        "models/enrollment_model.pkl"
    )
    revenue_model = joblib.load(
        "models/revenue_model.pkl"
    )
    metadata = joblib.load(
        "models/model_metadata.pkl"
    )

    return (
        enrollment_model,
        revenue_model,
        metadata
    )
@st.cache_data
def load_data():

    model_data = pd.read_csv(
        "modeling_dataset.csv"
    )

    category_data = pd.read_csv(
        "category_analysis.csv"
    )

    feature_data = pd.read_csv(
        "feature_importance.csv"
    )

    return (
        model_data,
        category_data,
        feature_data
    )

enrollment_model, revenue_model, metadata = load_models()

model_data, category_data, feature_data = load_data()

st.title(
    "📚 EduPro Predictive Analytics Dashboard"
)

st.markdown(
    """
    ### Course Demand & Revenue Forecasting

    This dashboard uses machine learning to forecast
    future course enrollments and revenue.
    """
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dashboard",
        "Course Prediction",
        "Category Analysis",
        "Feature Importance"
    ]
)

if page == "Dashboard":

    st.header(
        "📊 EduPro Business Overview"
    )
    total_courses = model_data.shape[0]

    total_enrollments = (
        model_data["PastEnrollments"].sum()
    )
    total_revenue = (
        model_data["PastRevenue"].sum()
    )
    avg_rating = (
        model_data["CourseRating"].mean()
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Total Courses",
        total_courses
    )

    col2.metric(
        "Historical Enrollments",
        f"{total_enrollments:,.0f}"
    )
    col3.metric(
        "Historical Revenue",
        f"₹{total_revenue:,.0f}"
    )
    col4.metric(
        "Average Course Rating",
        f"{avg_rating:.2f}"
    )
    st.divider()

    st.subheader(
        "Course Category Performance"
    )
    fig = px.bar(
        category_data,
        x="CourseCategory",
        y="Enrollments",
        title="Enrollments by Category"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )
    fig2 = px.bar(
        category_data,
        x="CourseCategory",
        y="Revenue",
        title="Revenue by Category"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

elif page == "Course Prediction":

    st.header(
        "🔮 Course Demand & Revenue Prediction"
    )
    st.write(
        "Enter course and instructor characteristics."
    )
    col1, col2 = st.columns(2)
    with col1:

        course_category = st.selectbox(
            "Course Category",
            sorted(
                model_data[
                    "CourseCategory"
                ].dropna().unique()
            )
        )
        course_type = st.selectbox(
            "Course Type",
            sorted(
                model_data[
                    "CourseType"
                ].dropna().unique()
            )
        )
        course_level = st.selectbox(
            "Course Level",
            sorted(
                model_data[
                    "CourseLevel"
                ].dropna().unique()
            )
        )
        course_price = st.number_input(
            "Course Price",
            min_value=0.0,
            max_value=1000.0,
            value=200.0
        )
        course_duration = st.number_input(
            "Course Duration",
            min_value=1.0,
            max_value=100.0,
            value=20.0
        )
        course_rating = st.slider(
            "Course Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )
    with col2:

        expertise = st.selectbox(
            "Instructor Expertise",
            sorted(
                model_data[
                    "Expertise"
                ].dropna().unique()
            )
        )
        years_experience = st.number_input(
            "Instructor Experience",
            min_value=0,
            max_value=40,
            value=5
        )
        teacher_rating = st.slider(
            "Instructor Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )
        past_enrollments = st.number_input(
            "Historical Enrollments",
            min_value=0,
            value=100
        )
        past_revenue = st.number_input(
            "Historical Revenue",
            min_value=0.0,
            value=10000.0
        )
        past_average_revenue = st.number_input(
            "Historical Average Revenue",
            min_value=0.0,
            value=100.0
        )
        past_revenue_std = st.number_input(
            "Historical Revenue Std",
            min_value=0.0,
            value=100.0
        )
        unique_users = st.number_input(
            "Historical Unique Users",
            min_value=0,
            value=80
        )

    if course_price == 0:
        price_band = "Free"

    elif course_price <= 150:
        price_band = "Low"

    elif course_price <= 300:
        price_band = "Medium"

    else:
        price_band = "High"

    if course_duration <= 10:
        duration_bucket = "Short"

    elif course_duration <= 25:
        duration_bucket = "Medium"

    else:
        duration_bucket = "Long"


    if course_rating < 3:
        rating_tier = "Low"

    elif course_rating < 4:
        rating_tier = "Medium"

    else:
        rating_tier = "High"


    if years_experience <= 3:
        experience_bucket = "Junior"

    elif years_experience <= 7:
        experience_bucket = "Mid"

    else:
        experience_bucket = "Senior"

    def calculate_expertise_match(
        category,
        expertise
    ):
        category = category.lower()
        expertise = expertise.lower()
        keywords = {

            "programming":
            ["programming", "software", "computer"],

            "data science":
            ["data", "analytics", "statistics"],

            "machine learning":
            ["machine", "learning", "ai"],

            "artificial intelligence":
            ["artificial", "intelligence", "ai"],

            "web development":
            ["web", "development"],

            "cybersecurity":
            ["security", "cyber"],

            "design":
            ["design", "ui", "ux"],

            "marketing":
            ["marketing"],

            "digital marketing":
            ["marketing", "digital"],

            "finance":
            ["finance", "accounting"],

            "business":
            ["business", "management"],

            "project management":
            ["management", "project"]
        }
        terms = keywords.get(
            category,
            []
        )
        return int(
            any(
                term in expertise
                for term in terms
            )
        )

    expertise_match = calculate_expertise_match(
        course_category,
        expertise
    )

    input_data = pd.DataFrame({
        "CourseCategory": [
            course_category
        ],
        "CourseType": [
            course_type
        ],
        "CourseLevel": [
            course_level
        ],

        "CoursePrice": [
            course_price
        ],
        "CourseDuration": [
            course_duration
        ],

        "CourseRating": [
            course_rating
        ],
        "Expertise": [
            expertise
        ],

        "YearsOfExperience": [
            years_experience
        ],

        "TeacherRating": [
            teacher_rating
        ],
        "PriceBand": [
            price_band
        ],

        "DurationBucket": [
            duration_bucket
        ],

        "RatingTier": [
            rating_tier
        ],

        "ExperienceBucket": [
            experience_bucket
        ],

        "ExpertiseMatch": [
            expertise_match
        ],
        "PastEnrollments": [
            past_enrollments
        ],

        "PastRevenue": [
            past_revenue
        ],

        "PastAverageRevenue": [
            past_average_revenue
        ],
        "PastRevenueStd": [
            past_revenue_std
        ],

        "UniqueUsers": [
            unique_users
        ]
    })

    if st.button(
        "🚀 Predict Course Performance",
        type="primary"
    ):
        predicted_enrollment = (
            enrollment_model.predict(
                input_data
            )[0]
        )
        predicted_revenue = (
            revenue_model.predict(
                input_data
            )[0]
        )
        predicted_enrollment = max(
            0,
            predicted_enrollment
        )
        predicted_revenue = max(
            0,
            predicted_revenue
        )
        st.success(
            "Prediction completed!"
        )
        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Enrollments",
            f"{predicted_enrollment:,.0f}"
        )
        c2.metric(
            "Predicted Revenue",
            f"₹{predicted_revenue:,.2f}"
        )

elif page == "Category Analysis":

    st.header(
        "📈 Category-Level Analysis"
    )
    metric = st.selectbox(
        "Select Metric",
        [
            "Enrollments",
            "Revenue",
            "AverageRevenue"
        ]
    )
    fig = px.bar(
        category_data,
        x="CourseCategory",
        y=metric,
        title=f"{metric} by Course Category"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        category_data,
        use_container_width=True
    )

elif page == "Feature Importance":

    st.header(
        "🎯 Feature Importance Explorer"
    )

    top_n = st.slider(
        "Number of features",
        min_value=5,
        max_value=30,
        value=15
    )

    top_features = (
        feature_data
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(top_n)
    )

    fig = px.bar(
        top_features,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Most Important Predictive Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        top_features,
        use_container_width=True
    )

st.sidebar.divider()

st.sidebar.info(
    f"""
    **EduPro Predictive Analytics**

    Enrollment Model:
    {metadata['best_enrollment_model']}

    Revenue Model:
    {metadata['best_revenue_model']}
    """
)
