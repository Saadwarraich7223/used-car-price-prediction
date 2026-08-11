"""Streamlit demo app for used car price prediction.

Run with:  streamlit run app.py
"""

import streamlit as st

from src.models.predict import predict_price

FUEL_OPTIONS = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
SELLER_OPTIONS = ["Individual", "Dealer", "Trustmark Dealer"]
TRANSMISSION_OPTIONS = ["Manual", "Automatic"]
OWNER_OPTIONS = [
    "First Owner",
    "Second Owner",
    "Third Owner",
    "Fourth & Above Owner",
    "Test Drive Car",
]


def format_inr(amount: float) -> str:
    """Format a rupee amount the Indian way (Lakh/Crore units)."""
    if amount >= 10_000_000:
        return f"Rs {amount / 10_000_000:.2f} Crore"
    if amount >= 100_000:
        return f"Rs {amount / 100_000:.2f} Lakh"
    return f"Rs {int(round(amount)):,}"


def get_prediction(car: dict) -> float:
    """Run the model on a raw car dict and return the predicted price."""
    return predict_price(car)


def main() -> None:
    st.set_page_config(page_title="Used Car Price Predictor", page_icon="car")
    st.title("Used Car Price Predictor")
    st.caption("Predict the resale price of a used car with a trained Random Forest model.")

    with st.form("car_form"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Car name", value="Maruti Swift Dzire VDI")
            year = st.number_input("Year of purchase", min_value=1985, max_value=2025, value=2014)
            km_driven = st.number_input("Kilometers driven", min_value=0, value=145500)
            fuel = st.selectbox("Fuel type", FUEL_OPTIONS)
            seller_type = st.selectbox("Seller type", SELLER_OPTIONS)
            transmission = st.selectbox("Transmission", TRANSMISSION_OPTIONS)
            owner = st.selectbox("Owner type", OWNER_OPTIONS)

        with col2:
            mileage = st.text_input("Mileage (e.g. 23.4 kmpl)", value="23.4 kmpl")
            engine = st.text_input("Engine (e.g. 1248 CC)", value="1248 CC")
            max_power = st.text_input("Max power (e.g. 74 bhp)", value="74 bhp")
            torque = st.text_input("Torque (e.g. 190Nm@ 2000rpm)", value="190Nm@ 2000rpm")
            seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])

        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        car = {
            "name": name,
            "year": int(year),
            "km_driven": int(km_driven),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "torque": torque,
            "seats": int(seats),
        }
        try:
            price = get_prediction(car)
        except FileNotFoundError as exc:
            st.error(f"Model artifact missing. Train it first with: `python -m src.models.train`")
            st.caption(str(exc))
        else:
            st.success(f"Estimated selling price: **{format_inr(price)}**")


if __name__ == "__main__":
    main()
