import csv
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def get_price_prediction(
    room_type_entire_home_apt,
    room_type_private_room,
    accommodates,
    imp_bathrooms,
    imp_bedrooms,
    imp_beds,
    cable_tv,
    tv,
    accommodates_bin_5_6,
    bedrooms_bin_1_2,
    bedrooms_bin_3_4,
    bedrooms_bin_4_6,
    beds_bin_6_12,
    city_DC,
    city_NYC,
    city_SF,
):

    price = (
        -63.0449
        + 89.9240 * room_type_entire_home_apt
        + 18.4589 * room_type_private_room
        + 14.6678 * accommodates
        + 45.3183 * imp_bathrooms
        + 29.9175 * imp_bedrooms
        - 7.2783  * imp_beds
        + 13.1935 * cable_tv
        + 9.5330  * tv
        + 9.4828  * accommodates_bin_5_6
        - 7.2843  * bedrooms_bin_1_2
        + 47.9075 * bedrooms_bin_3_4
        + 45.4899 * bedrooms_bin_4_6
        - 32.4389 * beds_bin_6_12
        + 39.5146 * city_DC
        + 23.9878 * city_NYC
        + 76.3766 * city_SF
    )
    print(f"Predicted Price: {price}")
    return price

def convert_na_cells_to_num(col_name, df, measure_type):
    # Create two new column names based on original column name.
    indicator_col_name = "m_" + col_name  # Tracks whether imputed.
    imputed_col_name = "imp_" + col_name  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputed_value = 0
    if measure_type == "median":
        imputed_value = df[col_name].median()
    elif measure_type == "mode":
        imputed_value = float(df[col_name].mode())
    else:
        imputed_value = df[col_name].mean()

    # Populate new columns with data.
    imputed_column = []
    indicator_column = []
    for i in range(len(df)):
        is_imputed = False

        # mi_OriginalName column stores imputed & original data.
        if np.isnan(df.loc[i][col_name]):
            is_imputed = True
            imputed_column.append(imputed_value)
        else:
            imputed_column.append(df.loc[i][col_name])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if is_imputed:
            indicator_column.append(1)
        else:
            indicator_column.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicator_col_name] = indicator_column
    df[imputed_col_name] = imputed_column
    return df

def load_data(df):
    # Impute the numeric columns used in model
    if "bathrooms" in df.columns and "imp_bathrooms" not in df.columns:
        df = convert_na_cells_to_num("bathrooms", df, "mode")
    if "review_scores_rating" in df.columns and "imp_review_scores_rating" not in df.columns:
        df = convert_na_cells_to_num("review_scores_rating", df, "median")
    if "bedrooms" in df.columns and "imp_bedrooms" not in df.columns:
        df = convert_na_cells_to_num("bedrooms", df, "mode")
    if "beds" in df.columns and "imp_beds" not in df.columns:
        df = convert_na_cells_to_num("beds", df, "mean")

    # city bucketing
    df["city"] = df["city"].where(df["city"].isin(["LA", "SF", "NYC", "DC", "Chicago"]), "Other")

    # outlier treatment
    for col in ["imp_bedrooms", "imp_beds", "imp_bathrooms"]:
        lo = df[col].quantile(0.01)
        df[col] = df[col].clip(lower=lo)
    hi_acc = df["accommodates"].quantile(0.99)
    df["accommodates"] = df["accommodates"].clip(upper=hi_acc)

    # dummies
    d_room = pd.get_dummies(df["room_type"].astype(str), prefix="room_type")
    for k in ["room_type_Entire home/apt", "room_type_Private room"]:
        if k not in d_room.columns: d_room[k] = 0

    d_city = pd.get_dummies(df["city"], prefix="city")
    for k in ["city_DC", "city_NYC", "city_SF"]:
        if k not in d_city.columns: d_city[k] = 0

    # bins
    acc_bins = pd.cut(df["accommodates"], bins=[0, 2, 3, 4, 5, 6, 8, 16, 1000])
    d_acc = pd.get_dummies(acc_bins, prefix="accommodatesBin")
    if "accommodatesBin_(5, 6]" not in d_acc.columns:
        d_acc["accommodatesBin_(5, 6]"] = 0

    bed_bins = pd.cut(df["imp_bedrooms"], bins=[-1, 1, 2, 3, 4, 6, 10, 1000])
    d_bed = pd.get_dummies(bed_bins, prefix="bedroomsBin")
    for k in ["bedroomsBin_(1, 2]", "bedroomsBin_(3, 4]", "bedroomsBin_(4, 6]"]:
        if k not in d_bed.columns: d_bed[k] = 0

    beds_bins = pd.cut(df["imp_beds"], bins=[-1, 1, 2, 3, 4, 6, 12, 1000])
    d_beds = pd.get_dummies(beds_bins, prefix="bedsBin")
    if "bedsBin_(6, 12]" not in d_beds.columns:
        d_beds["bedsBin_(6, 12]"] = 0

    X = pd.DataFrame({
        "room_type_Entire home/apt": d_room["room_type_Entire home/apt"].astype(int),
        "room_type_Private room": d_room["room_type_Private room"].astype(int),
        "accommodates": df["accommodates"].astype(float),
        "imp_bathrooms": df["imp_bathrooms"].astype(float),
        "imp_bedrooms": df["imp_bedrooms"].astype(float),
        "imp_beds": df["imp_beds"].astype(float),
        "cable_tv": df["cable_tv"].astype(int),
        "tv": df["tv"].astype(int),
        "accommodatesBin_(5, 6]": d_acc["accommodatesBin_(5, 6]"].astype(int),
        "bedroomsBin_(1, 2]": d_bed["bedroomsBin_(1, 2]"].astype(int),
        "bedroomsBin_(3, 4]": d_bed["bedroomsBin_(3, 4]"].astype(int),
        "bedroomsBin_(4, 6]": d_bed["bedroomsBin_(4, 6]"].astype(int),
        "bedsBin_(6, 12]": d_beds["bedsBin_(6, 12]"].astype(int),
        "city_DC": d_city["city_DC"].astype(int),
        "city_NYC": d_city["city_NYC"].astype(int),
        "city_SF": d_city["city_SF"].astype(int),
        })

    return X


def main():
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        predictions = []

        try:

            df = pd.read_csv("AirBNB_mystery.csv")
            X = load_data(df)
            V = X.values

            for i in range(len(V)):
                predictions.append(
                    get_price_prediction(
                        V[i][0],  # room_type_Entire home/apt
                        V[i][1],  # room_type_Private room
                        V[i][2],  # accommodates
                        V[i][3],  # imp_bathrooms
                        V[i][4],  # imp_bedrooms
                        V[i][5],  # imp_beds
                        V[i][6],  # cable_tv
                        V[i][7],  # tv
                        V[i][8],  # accommodatesBin_(5, 6]
                        V[i][9],  # bedroomsBin_(1, 2]
                        V[i][10],  # bedroomsBin_(3, 4]
                        V[i][11],  # bedroomsBin_(4, 6]
                        V[i][12],  # bedsBin_(6, 12]
                        V[i][13],  # city_DC
                        V[i][14],  # city_NYC
                        V[i][15],  # city_SF
                    )
                )
        except Exception as e:
            print("Failed reading or preparing mystery file:", e)

        try:
            with open("AirBNB_predictions.csv", "w", encoding="UTF8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["Predicted Price"])
                for p in predictions:
                    w.writerow([float(p)])
        except Exception as e:
            print("Failed writing output CSV:", e)


if __name__ == "__main__":
        main()