import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
DEFAULT_FILE_PATH = "met_office_data.csv"


# Utility functions
def get_k_value(wind_speed: float, cloud_cover: float) -> Optional[float]:
    """Determine the value of K based on wind speed and cloud cover."""
    try:
        if wind_speed <= 12:
            if cloud_cover <= 2:
                return -2.2
            elif cloud_cover <= 4:
                return -1.7
            elif cloud_cover <= 6:
                return -0.6
            else:
                return 0
        elif wind_speed <= 25:
            if cloud_cover <= 2:
                return -1.1
            elif cloud_cover <= 4:
                return 0
            elif cloud_cover <= 6:
                return 0.6
            else:
                return 1.1
        elif wind_speed <= 38:
            if cloud_cover <= 2:
                return -0.6
            elif cloud_cover <= 4:
                return 0
            elif cloud_cover <= 6:
                return 0.6
            else:
                return 1.1
        elif wind_speed <= 51:
            if cloud_cover <= 2:
                return 1.1
            elif cloud_cover <= 4:
                return 1.7
            elif cloud_cover <= 6:
                return 2.8
            else:
                return 3.5
        else:
            return None
    except Exception as e:
        logging.error(f"Error calculating K value: {e}")
        return None


def calculate_min_temp(midday_temp: float, dew_point: float, wind_speed: float, cloud_cover: float) -> Optional[float]:
    """Calculate the overnight minimum temperature using the corrected formula."""
    k = get_k_value(wind_speed, cloud_cover)
    if k is None:
        logging.warning("K value is None; cannot calculate minimum temperature.")
        return None
    return 0.316 * midday_temp + 0.548 * dew_point - 1.24 + k


def calculate_actual_min_temp(midday_temp: float, dew_point: float, k: float) -> float:
    """Calculate the actual minimum temperature using the McKenzie method."""
    return 0.5 * (midday_temp + dew_point) - k


def calculate_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate error metrics and add them to the DataFrame."""
    df["Absolute Error"] = abs(df["Overnight Min Temp"] - df["Actual Minimum Temperature"])
    df["Squared Error"] = (df["Overnight Min Temp"] - df["Actual Minimum Temperature"]) ** 2
    return df


def plot_results(df: pd.DataFrame):
    """Plot actual vs predicted temperatures and error metrics."""
    # Plot actual vs predicted temperatures
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df["Actual Minimum Temperature"], label="Actual Minimum Temperature", marker="o")
    plt.plot(df.index, df["Overnight Min Temp"], label="Predicted Minimum Temperature", marker="x")
    plt.xlabel("Data Point Index")
    plt.ylabel("Temperature (°C)")
    plt.title("Actual vs Predicted Minimum Temperature")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot error metrics
    mae = df["Absolute Error"].mean()
    mse = df["Squared Error"].mean()
    rmse = np.sqrt(mse)
    error_metrics = [mae, mse, rmse]
    error_labels = ["MAE", "MSE", "RMSE"]

    plt.figure(figsize=(8, 5))
    plt.bar(error_labels, error_metrics, color=['blue', 'orange', 'green'])
    plt.xlabel("Error Metric")
    plt.ylabel("Error Value")
    plt.title("Error Metrics: MAE, MSE, RMSE")
    plt.show()


def main(file_path: str = DEFAULT_FILE_PATH):
    """Main function to run the analysis pipeline."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    try:
        # Load data
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully.")

        # Calculate minimum temperatures
        df["Overnight Min Temp"] = df.apply(
            lambda row: calculate_min_temp(row["Midday_Temperature"], row["Midday_Dew_Point"], row["Wind_Speed"],
                                           row["Cloud_Cover"]),
            axis=1
        )

        df["Actual Minimum Temperature"] = df.apply(
            lambda row: calculate_actual_min_temp(row["Midday_Temperature"], row["Midday_Dew_Point"],
                                                  get_k_value(row["Wind_Speed"], row["Cloud_Cover"])),
            axis=1
        )

        # Calculate errors
        df = calculate_errors(df)

        # Calculate metrics
        mae = df["Absolute Error"].mean()
        mse = df["Squared Error"].mean()
        rmse = np.sqrt(mse)

        logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
        logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Plot results
        plot_results(df)

    except Exception as e:
        logging.error(f"Error in main pipeline: {e}")


if __name__ == "__main__":
    main()
