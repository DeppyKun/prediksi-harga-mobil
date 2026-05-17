import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


st.set_page_config(
    page_title="Prediksi Harga Mobil",
    page_icon="🚗",
    layout="wide"
)


@st.cache_data
def load_data():
    try:
        data = pd.read_excel("Car_sales.xls")
    except Exception:
        data = pd.read_csv("Car_sales.xls")
    return data


@st.cache_resource
def train_model():
    df = load_data()

    # Menghapus data yang target harganya kosong
    df = df.dropna(subset=["Price_in_thousands"]).copy()

    numeric_features = [
        "__year_resale_value",
        "Engine_size",
        "Horsepower",
        "Wheelbase",
        "Width",
        "Length",
        "Curb_weight",
        "Fuel_capacity",
        "Fuel_efficiency"
    ]

    categorical_features = ["Vehicle_type"]

    features = numeric_features + categorical_features

    X = df[features]
    y = df["Price_in_thousands"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    evaluation = {
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    }

    return model, evaluation, df


model, evaluation, df = train_model()


st.title("🚗 Aplikasi Prediksi Harga Mobil")
st.write(
    """
    Aplikasi ini digunakan untuk memprediksi harga mobil berdasarkan spesifikasi kendaraan.
    Model yang digunakan adalah **Linear Regression**.
    """
)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RMSE", f"{evaluation['RMSE']:.4f}")
with col2:
    st.metric("MAE", f"{evaluation['MAE']:.4f}")
with col3:
    st.metric("R2 Score", f"{evaluation['R2 Score']:.4f}")

st.info(
    """
    Keterangan:
    - RMSE dan MAE menggunakan satuan ribu dolar.
    - R2 Score menunjukkan kemampuan model dalam menjelaskan variasi harga mobil.
    """
)

st.divider()

st.header("Input Spesifikasi Mobil")

left, right = st.columns(2)

with left:
    resale_value = st.number_input(
        "Year Resale Value",
        min_value=0.0,
        value=13.2275,
        step=0.1
    )

    engine_size = st.number_input(
        "Engine Size",
        min_value=0.0,
        value=2.45,
        step=0.1
    )

    horsepower = st.number_input(
        "Horsepower",
        min_value=0.0,
        value=142.50,
        step=1.0
    )

    wheelbase = st.number_input(
        "Wheelbase",
        min_value=0.0,
        value=110.05,
        step=0.1
    )

    width = st.number_input(
        "Width",
        min_value=0.0,
        value=70.25,
        step=0.1
    )

with right:
    length = st.number_input(
        "Length",
        min_value=0.0,
        value=189.75,
        step=0.1
    )

    curb_weight = st.number_input(
        "Curb Weight",
        min_value=0.0,
        value=3.227,
        step=0.1
    )

    fuel_capacity = st.number_input(
        "Fuel Capacity",
        min_value=0.0,
        value=19.25,
        step=0.1
    )

    fuel_efficiency = st.number_input(
        "Fuel Efficiency",
        min_value=0.0,
        value=24.00,
        step=1.0
    )

    vehicle_type = st.selectbox(
        "Vehicle Type",
        ["Passenger", "Car"]
    )


input_data = pd.DataFrame({
    "__year_resale_value": [resale_value],
    "Engine_size": [engine_size],
    "Horsepower": [horsepower],
    "Wheelbase": [wheelbase],
    "Width": [width],
    "Length": [length],
    "Curb_weight": [curb_weight],
    "Fuel_capacity": [fuel_capacity],
    "Fuel_efficiency": [fuel_efficiency],
    "Vehicle_type": [vehicle_type]
})

st.subheader("Data Input")
st.dataframe(input_data, use_container_width=True)

if st.button("Prediksi Harga Mobil"):
    prediction = model.predict(input_data)

    price_thousand = prediction[0]
    price_dollar = price_thousand * 1000

    st.success(f"Prediksi harga mobil: ${price_dollar:,.2f}")

    st.write("### Ringkasan Hasil")
    st.write(f"Prediksi harga dalam ribu dolar: **{price_thousand:.4f}**")
    st.write(f"Prediksi harga dalam dolar: **${price_dollar:,.2f}**")

    st.write("### Interpretasi")
    st.write(
        """
        Berdasarkan spesifikasi yang dimasukkan, sistem memperkirakan harga mobil
        menggunakan model Linear Regression. Hasil prediksi ini dapat digunakan
        sebagai rekomendasi awal dalam menentukan harga kendaraan baru.
        """
    )

st.divider()

st.header("Informasi Dataset")

st.write("Jumlah data:", df.shape[0])
st.write("Jumlah kolom:", df.shape[1])

with st.expander("Lihat Dataset"):
    st.dataframe(df, use_container_width=True)

with st.expander("10 Mobil dengan Penjualan Terbanyak"):
    df_top = df.copy()
    df_top["Car_Name"] = df_top["Manufacturer"] + " " + df_top["Model"]
    top10 = df_top.sort_values(
        by="Sales_in_thousands",
        ascending=False
    ).head(10)

    st.dataframe(
        top10[[
            "Car_Name",
            "Sales_in_thousands",
            "Price_in_thousands",
            "Vehicle_type",
            "Engine_size",
            "Horsepower",
            "Fuel_efficiency"
        ]],
        use_container_width=True
    )

    st.bar_chart(
        top10.set_index("Car_Name")["Sales_in_thousands"]
    )

st.caption("Aplikasi dibuat untuk Final Project Sains Data - Prediksi Harga Mobil.")
st.caption("Guna Pandu Prilana")
st.caption("237006053")
