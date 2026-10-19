
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Chennai Weather Prediction",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Chennai Weather Prediction")
st.write(
    "Daily temperature analysis and prediction using Python "
    "and machine learning."
)

st.info(
    "Upload your temperature CSV to explore historical data "
    "and generate a simple next-day baseline estimate."
)

uploaded_file = st.file_uploader(
    "Upload your Chennai temperature dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        required_columns = ["Date", "Temperature_C"]

        if not all(c in df.columns for c in required_columns):
            st.error(
                "CSV must contain Date and Temperature_C columns."
            )
        else:
            df["Date"] = pd.to_datetime(
                df["Date"], errors="coerce"
            )
            df["Temperature_C"] = pd.to_numeric(
                df["Temperature_C"], errors="coerce"
            )

            df = (
                df.dropna(subset=required_columns)
                .sort_values("Date")
                .drop_duplicates(subset=["Date"], keep="last")
            )

            if df.empty:
                st.error("No valid temperature records found.")
            else:
                latest = df.iloc[-1]
                temperature = float(latest["Temperature_C"])

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Latest Temperature",
                    f"{temperature:.2f} °C"
                )
                col2.metric(
                    "Latest Recorded Date",
                    latest["Date"].strftime("%d %b %Y")
                )
                col3.metric(
                    "Next-Day Baseline",
                    f"{temperature:.2f} °C"
                )

                st.caption(
                    "The baseline assumes tomorrow's temperature "
                    "equals the latest recorded temperature. "
                    "It is not a live weather forecast."
                )

                st.subheader("Temperature Trend")

                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(
                    df["Date"],
                    df["Temperature_C"],
                    label="Daily Temperature"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel("Temperature (°C)")
                ax.set_title("Chennai Daily Temperature History")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Dataset Preview")
                st.dataframe(df.tail(20), use_container_width=True)

                csv_data = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download cleaned dataset",
                    data=csv_data,
                    file_name="chennai_temperature_cleaned.csv",
                    mime="text/csv"
                )

    except Exception as error:
        st.error(f"Could not process the file: {error}")
else:
    st.write(
        "Expected CSV columns: Date, Temperature_C"
    )
