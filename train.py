
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from mlxtend.classifier import LogisticRegression
from mlxtend.evaluate import confusion_matrix, accuracy_score
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import RFE, f_regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model    import LinearRegression
from sklearn                 import metrics
import statsmodels.api       as sm
import numpy                 as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Import data into a DataFrame.
def get_data():
    path = "/Users/nataliaarseniuk/PycharmProjects/PythonProject/assignment1/"
    csv = "AirBNB.csv"
    df = pd.read_csv(path + csv)
    return df


def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_'   + colName # Tracks whether imputed.
    imputedColName   = 'imp_' + colName # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if(measureType=="median"):
        imputedValue = df[colName].median()
    elif(measureType=="mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn  = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if(np.isnan(df.loc[i][colName])):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.loc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if(isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName]   = imputedColumn
    return df

def display_histograms(df):
    # Plot histogram of all columns.
    df.hist(bins=50, figsize=(20, 15))
    plt.show()


def display_scatter_matrix(df):
    # Scatter plot of all columns.
    scatter_matrix(df, figsize=(20, 15))
    plt.show()

from sklearn.model_selection import KFold
def cross_fold_validation(X, y, NUM_SPLITS):
    cv = KFold(n_splits=NUM_SPLITS, shuffle=True)
    rmses = []

    for train_indices, test_indices in cv.split(X):
        # Slice using iloc since cv.split() gives positional indices
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        # build model
        model = LinearRegression()
        model.fit(X_train,y_train)

        predictions = model.predict(X_test)
        mse         = mean_squared_error(predictions, y_test)
        rmse        = np.sqrt(mse)
        print("RMSE: " + str(rmse))
        rmses.append(rmse)

    avgRMSE = np.mean(rmses)
    print("Average rmse: " + str(avgRMSE))

def plot_prediction_vs_actual(title, y_test, predictions):
    plt.scatter(y_test, predictions)
    plt.legend()
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted (Y) vs. Actual (X): " + title)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--")

def plot_residuals_vs_actual(title, y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(y_test, residuals, label="Residuals vs Actual")
    plt.xlabel("Actual")
    plt.ylabel("Residual")
    plt.title("Error Residuals (Y) vs. Actual (X): " + title)
    plt.plot([y_test.min(), y_test.max()], [0, 0], "k--")

def plot_residual_histogram(title, y_test, predictions, bins):
    residuals = y_test - predictions
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.hist(residuals, label="Residuals vs Actual", bins=bins)
    plt.title("Error Residual Frequency: " + title)


def draw_validation_plots(title, bins, y_test, predictions):
    # Define number of rows and columns for graph display.
    plt.subplots(nrows=1, ncols=3, figsize=(20, 5))

    plt.subplot(1, 3, 1)  # Specify total rows, columns and image #
    plot_prediction_vs_actual(title, y_test, predictions)

    plt.subplot(1, 3, 2)  # Specify total rows, columns and image #
    plot_residuals_vs_actual(title, y_test, predictions)

    plt.subplot(1, 3, 3)  # Specify total rows, columns and image #
    plot_residual_histogram(title, y_test, predictions, bins)
    plt.show()


def view_and_get_outliers(df, col_name, threshold):
    # Show basic statistics.
    df_sub = df[[col_name]]
    print("*** Statistics for " + col_name)
    print(df_sub.describe())

    # Show boxplot.
    df_sub.boxplot(column=[col_name])
    plt.title(col_name)
    plt.show()

    # Note this is absolute 'abs' so it gets both high and low values.
    z = np.abs(stats.zscore(df_sub))
    row_column_array = np.where(z > threshold)
    row_indices = row_column_array[0]

    # Show outlier rows.
    print("\nOutlier row indexes for " + col_name + ":")
    print(row_indices)
    print("")

    # Show filtered and sorted DataFrame with outliers.
    df_sub = df.iloc[row_indices]
    print("\nDataFrame rows containing outliers for " + col_name + ":")
    print(df_sub.sort_values([col_name], ascending=[True]))
    return row_indices



# Imputation by converting NA cells to median, mode, or mean.
# forward feature selection
def model_a():
    df = get_data()
    print(df.info())
    print(df.describe())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convertNAcellsToNum("bathrooms", df, "mode")
    df = convertNAcellsToNum("review_scores_rating", df, "median")
    df = convertNAcellsToNum("bedrooms", df, "mode")
    df = convertNAcellsToNum("beds", df, "mean")

    X = df[
        [
                #"property_type", non-numeric
                #"room_type", non-numeric
                "accommodates",
                #"m_bathrooms", #insignificant
                "imp_bathrooms",
                # #"bed_type", non-numeric
                # #"cancellation_policy", non-numeric
                # #"cleaning_fee", non-numeric
                # #"city", non-numeric
                # #"first_review", non-numeric
                # #"host_has_profile_pic", non-numeric
                # #"host_identity_verified", non-numeric
                # #"host_response_rate", non-numeric
                # #"host_since", non-numeric
                # #"instant_bookable", non-numeric
                # #"last_review", non-numeric
                # #"neighbourhood", non-numeric
                # "number_of_reviews",
                # "m_review_scores_rating",
                # "imp_review_scores_rating",
                # #"zipcode", non-numeric
                # "m_bedrooms",
                "imp_bedrooms",
                # "m_beds",
                "imp_beds",
                # "24_hour_check_in",
                # "accessible_height_bed",
                # "accessible_height_toilet",
                # "air_conditioning",
                # "air_purifier",
                # "bbq_grill",
                # "baby_bath",
                # "baby_monitor",
                # "babysitter_recommendations",
                # "bath_towel",
                # "bathtub",
                # "bathtub_with_shower_chair", #insignificant
                # "beach_essentials", #insignificant
                # "beachfront", #insignificant
                # "bed_linens", #insignificant
                # "body_soap", #insignificant
                # "breakfast", #insignificant
                # "buzzer_wireless_intercom",
                "cable_tv",
                # "carbon_monoxide_detector",
                # "cat_s", #insignificant
                # "changing_table", #insignificant
                # "children’s_books_and_toys", #insignificant
                # "children’s_dinnerware", #insignificant
                # "cleaning_before_checkout", #insignificant
                # "coffee_maker", #insignificant
                # "cooking_basics", #insignificant
                # "crib",
                # "disabled_parking_spot", #insignificant
                # "dishes_and_silverware", #insignificant
                # "dishwasher",
                # "dog_s", #insignificant
                # "doorman",
                # "doorman_entry",
                # "dryer", #insignificant
                # "ev_charger", #insignificant
                # "elevator",
                # "elevator_in_building", #insignificant
                # "essentials", #insignificant
                # "ethernet_connection", #insignificant
                # "extra_pillows_and_blankets", #insignificant
                #"family_kid_friendly",
                # "fire_extinguisher",
                # "fireplace_guards", #insignificant
                # "firm_matress", #insignificant
                # "firm_mattress", #insignificant
                # "first_aid_kit",
                # "fixed_grab_bars_for_shower_&_toilet", #insignificant
                # "flat_smooth_pathway_to_front_door", #insignificant
                # "flat_smooth_pathway_to_front_door.1", #insignificant
                # "free_parking_on_premises",
                # "free_parking_on_street", #insignificant
                # "game_console", #insignificant
                # "garden_or_backyard", #insignificant
                # "grab_rails_for_shower_and_toilet", #insignificant
                # "ground_floor_access", #insignificant
                # "gym",
                # "hair_dryer",
                # "hand_or_paper_towel", #insignificant
                # "hand_soap", #insignificant
                # "handheld_shower_head", #insignificant
                # "hangers",
                # "heating",
                # "high_chair", #insignificant
                # "host_greets_you", #insignificant
                # "hot_tub", #insignificant
                # "hot_water", #insignificant
                # "hot_water_kettle", #insignificant
                "indoor_fireplace",
                # "internet",
                # "iron", #insignificant
                # "keypad",
                # "kitchen", #insignificant
                # "lake_access", #insignificant
                # "laptop_friendly_workspace", #insignificant
                # "lock_on_bedroom_door",
                # "lockbox", #insignificant
                # "long_term_stays_allowed",  #insignificant
                # "luggage_dropoff_allowed", #insignificant
                # "microwave", #insignificant
                # "other", #insignificant
                # "other_pet_s", #insignificant
                # "outlet_covers", #insignificant
                # "oven", #insignificant
                # "pack_’n_play_travel_crib", #insignificant
                # "paid_parking_off_premises", #insignificant
                # "path_to_entrance_lit_at_night", #insignificant
                # "patio_or_balcony", #insignificant
                # "pets_allowed", #insignificant
                # "pets_live_on_this_property",
                # "pocket_wifi", #insignificant
                # "pool", #insignificant
                # "private_bathroom", #insignificant
                # "private_entrance",
                # "private_living_room",
                # "refrigerator", #insignificant
                # "roll_in_shower_with_chair", #insignificant
                # "room_darkening_shades", #insignificant
                # "safety_card", #insignificant
                # "self_check_in", #insignificant
                # "shampoo",
                # "single_level_home", #insignificant
                # "ski_in_ski_out", #insignificant
                # "smart_lock", #insignificant
                # "smartlock",
                # "smoke_detector", #insignificant
                # "smoking_allowed",
                # "stair_gates", #insignificant
                # "step_free_access", #insignificant
                # "stove", #insignificant
                # "suitable_for_events",
                "tv",
                # "table_corner_guards", #insignificant
                # "toilet_paper", #insignificant
                # "washer", #insignificant
                # "washer_dryer", #insignificant
                # "waterfront", #insignificant
                # "well_lit_path_to_entrance", #insignificant
                # "wheelchair_accessible", #insignificant
                # "wide_clearance_to_bed", #insignificant
                # "wide_clearance_to_shower_&_toilet", #insignificant
                # "wide_clearance_to_shower_and_toilet", #insignificant
                # "wide_doorway", #insignificant
                # "wide_entryway", #insignificant
                # "wide_hallway_clearance", #insignificant
                # "window_guards", #insignificant
                # "wireless_internet", #insignificant
                #"translation_missing:_en_hosting_amenity_49", #insignificant
        ]
    ]

    # used for selecting features
    # #  f_regression returns F statistic for each feature.
    # ffs = f_regression(X, y)
    #
    # featuresDf = pd.DataFrame()
    # for i in range(0, len(X.columns)):
    #     featuresDf = featuresDf._append({"feature": X.columns[i],
    #                                      "ffs": ffs[0][i]}, ignore_index=True)
    # featuresDf = featuresDf.sort_values(by=['ffs'])
    # print(featuresDf)
    # print("\nTop 10 Features Based on F-Regression:")
    # print(featuresDf.tail(10))


    # Adding an intercept *** This is required ***. Don't forget this step.
    # * This step is only needed when using sm.OLS. *
    # The intercept centers the error residuals around zero
    # which helps to avoid over-fitting.
    X = sm.add_constant(X)
    y = df['price']

    print("\nCross fold validation:")
    cross_fold_validation(X, y,3)

    y = df['price']

    # Create training set with 85% of data and test set with 15% of data.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test) # make the predictions by the model

    print(model.summary())

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    # Draw validation plots on the best model.
    draw_validation_plots(
        title="Model A: Price",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )


# imputation by converting NA cells to median, mode, or mean
# ffs
# binning and dummy variables
def model_b():
    df = get_data()
    print(df.info())
    print(df.describe())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convertNAcellsToNum("bathrooms", df, "mode")
    df = convertNAcellsToNum("review_scores_rating", df, "median")
    df = convertNAcellsToNum("bedrooms", df, "mode")
    df = convertNAcellsToNum("beds", df, "mean")

    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "property_type",
                        "room_type",
                        "bed_type",
                        "cancellation_policy",
                        "cleaning_fee",
                        # "city",
                        # "first_review",
                        "host_has_profile_pic",
                        "host_identity_verified",
                        # "host_response_rate",
                        # "host_since",
                        "instant_bookable",
                        # "last_review",
                        # "neighbourhood",
                        # "zipcode",
                        ]
                ],
                columns =[
                    "property_type",
                    "room_type",
                    "bed_type",
                    "cancellation_policy",
                    "cleaning_fee",
                    # "city",
                    # "first_review",
                    "host_has_profile_pic",
                    "host_identity_verified",
                    # "host_response_rate",
                    # "host_since",
                    "instant_bookable",
                    # "last_review",
                    # "neighbourhood",
                    # "zipcode",
                ],
            ).astype(int),
        ],
        axis=1,
    ) # Join dummy df with original df


    # Create bins.
    df["accommodatesBin"] = pd.cut(
        x=df["accommodates"],
        bins=[0, 2, 3, 4, 5, 6, 8, 16, 1000],
    )
    df["ratingBin"] = pd.cut(
        x=df["imp_review_scores_rating"],
        bins=[0, 60, 80, 90, 95, 100],
    )
    df["reviewsBin"] = pd.cut(
        x=df["number_of_reviews"],
        bins=[-1, 0, 10, 25, 50, 100, 300, 1000, 100000],
    )
    df["bathroomsBin"] = pd.cut(
        x=df["imp_bathrooms"],
        bins=[-1, 1, 2, 3, 4, 10, 1000],
    )
    df["bedroomsBin"] = pd.cut(
        x=df["imp_bedrooms"],
        bins=[-1, 1, 2, 3, 4, 6, 10, 1000],
    )
    df["bedsBin"] = pd.cut(
        x=df["imp_beds"],
        bins=[-1, 1, 2, 3, 4, 6, 12, 1000],
    )

    # Join dummies with original df
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "accommodatesBin",
                        "ratingBin",
                        "reviewsBin",
                        "bathroomsBin",
                        "bedroomsBin",
                        "bedsBin",
                    ]
                ],
                columns=[
                        "accommodatesBin",
                        "ratingBin",
                        "reviewsBin",
                        "bathroomsBin",
                        "bedroomsBin",
                        "bedsBin",
                ],
            ).astype(int),
        ],
        axis=1,
    )

    print("\nAfter imputation and dummy variables:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    # used for checking unique values
    # print(df["property_type"].unique())
    # print(df["room_type"].unique())
    # print(df["bed_type"].unique())
    # print(df["cancellation_policy"].unique())
    # print(df["cleaning_fee"].unique())
    # print(df["host_has_profile_pic"].unique())
    # print(df["host_identity_verified"].unique())
    # print(df["host_response_rate"].unique())
    # print(df["instant_bookable"].unique())


    x = df[
        [
            # "property_type", non-numeric, dummy
            # 'property_type_House',
            #'property_type_Apartment',
            # 'property_type_Other',
            # 'property_type_Loft',
            # 'property_type_Condominium',
            # 'property_type_Townhouse',
            # 'property_type_Cabin',
            # 'property_type_Dorm',
            # 'property_type_Bungalow',
            # 'property_type_Villa',
            # 'property_type_In-law',
            # 'property_type_Bed & Breakfast',
            # 'property_type_Guesthouse',
            # 'property_type_Tipi',
            # 'property_type_Boutique hotel',
            # 'property_type_Camper/RV',
            # 'property_type_Boat',
            # 'property_type_Hostel',
            #  'property_type_Guest suite',
            # 'property_type_Timeshare',
            # 'property_type_Yurt',
            # 'property_type_Serviced apartment',
            # 'property_type_Tent',
            # 'property_type_Treehouse',
            # 'property_type_Castle',
            # 'property_type_Vacation home',
            # 'property_type_Earth House',
            # 'property_type_Cave',
            # 'property_type_Hut',
            # 'property_type_Island',
            # 'property_type_Chalet',
            # 'property_type_Parking Space',
            # 'property_type_Casa particular',
            # 'property_type_Train',
            # "room_type", non-numeric, dummy
            'room_type_Entire home/apt',
            'room_type_Private room',
            # 'room_type_Shared room',

            "accommodates",
            "imp_bathrooms",
            #"bed_type", non-numeric, dummy
            # 'bed_type_Real Bed',
            # 'bed_type_Couch',
            # 'bed_type_Futon',
            # 'bed_type_Airbed',
            # 'bed_type_Pull-out Sofa',
            
            # "cancellation_policy", non-numeric, dummy
            #'cancellation_policy_strict',
            # 'cancellation_policy_moderate',
            # 'cancellation_policy_flexible',
            # 'cancellation_policy_super_strict_30',
            # 'cancellation_policy_super_strict_60',
            # "cleaning_fee", non-numeric, dummy
            # 'cleaning_fee_True',
            #'cleaning_fee_False',
            #"city", #non-numeric
            # "first_review", non-numeric
            # "host_has_profile_pic", non-numeric, dummy
            # 'host_has_profile_pic_t',
            #'host_has_profile_pic_f',
            # "host_identity_verified", non-numeric, dummy
            # 'host_identity_verified_t',
            #'host_identity_verified_f',
            # "host_response_rate", non-numeric
            # "host_since", non-numeric
            # "instant_bookable", non-numeric, dummy
            # "instant_bookable_f",
            #"instant_bookable_t",
            # "last_review", non-numeric
            # "neighbourhood", non-numeric
            # "number_of_reviews",
            #  "m_review_scores_rating",
            # "imp_review_scores_rating",
            "imp_bedrooms",
            "imp_beds",
            # "air_conditioning",
            # "bathtub",
            # "buzzer_wireless_intercom",
            "cable_tv",
            # "carbon_monoxide_detector",
            # "crib",
            # "dishwasher",
            # "doorman",
            # "doorman_entry",
            # "elevator",
            # "family_kid_friendly",
            # "fire_extinguisher",
            # "first_aid_kit",
            # "free_parking_on_premises",
            # "gym",
            # "hangers",
            # "heating",
            # "indoor_fireplace",
            # "internet",
            # "keypad",
            # "lock_on_bedroom_door",
            # "pets_live_on_this_property",
            # "private_entrance",
            # "private_living_room",
            # "shampoo",
            # "smartlock",
            # "smoking_allowed",
            #"suitable_for_events",
            "tv",
            # dummy
            #'accommodatesBin_(0, 2]',
            #'accommodatesBin_(2, 3]',
            # 'accommodatesBin_(3, 4]',
            # 'accommodatesBin_(4, 5]',
            'accommodatesBin_(5, 6]',
            #'accommodatesBin_(6, 8]',
            #'accommodatesBin_(8, 16]',
            # 'accommodatesBin_(16, 1000]',

            # 'ratingBin_(0, 60]',
            # 'ratingBin_(60, 80]',
            # #'ratingBin_(80, 90]',
            # 'ratingBin_(90, 95]',
            # 'ratingBin_(95, 100]',

            # 'reviewsBin_(-1, 0]',
            # 'reviewsBin_(0, 10]',
            # 'reviewsBin_(10, 25]',
            # 'reviewsBin_(25, 50]',
            # #'reviewsBin_(50, 100]',
            # 'reviewsBin_(100, 300]',
            # 'reviewsBin_(300, 1000]',
            # 'reviewsBin_(1000, 100000]',

            #'bathroomsBin_(-1, 1]',
            'bathroomsBin_(1, 2]',
            #'bathroomsBin_(2, 3]',
            # 'bathroomsBin_(3, 4]',
            'bathroomsBin_(4, 10]',
            # 'bathroomsBin_(10, 1000]',

            'bedroomsBin_(-1, 1]',
            'bedroomsBin_(1, 2]',
            'bedroomsBin_(2, 3]',
            'bedroomsBin_(3, 4]',
            'bedroomsBin_(4, 6]',
            # 'bedroomsBin_(6, 10]',
            # 'bedroomsBin_(10, 1000]',

            'bedsBin_(-1, 1]',
            # 'bedsBin_(1, 2]',
            #'bedsBin_(2, 3]',
            #'bedsBin_(3, 4]',
            #'bedsBin_(4, 6]',
            'bedsBin_(6, 12]',
            #'bedsBin_(12, 1000]',

        ]
    ]


    # used for top feature selection
    # #  f_regression returns F statistic for each feature.
    # ffs = f_regression(x, y)
    #
    # featuresDf = pd.DataFrame()
    # for i in range(0, len(x.columns)):
    #     featuresDf = featuresDf._append({"feature": x.columns[i],
    #                                      "ffs": ffs[0][i]}, ignore_index=True)
    # featuresDf = featuresDf.sort_values(by=['ffs'])
    # print(featuresDf)
    # print("\nTop 10 Features Based on F-Regression:")
    # print(featuresDf.tail(30))

    x = sm.add_constant(x)
    y = df["price"]

    print("\nCross fold validation:")
    cross_fold_validation(x, y, 3)

    y = df["price"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)
    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)

    print(model.summary(title="Model B: Price Prediction"))
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    draw_validation_plots(
        title="Model B: Price",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )

# imputation by converting NA cells to median, mode, or mean
# includes binning and dummy variables
# ffs
# also includes outlier treatment
def model_c():
    df = get_data()
    print(df.info())
    print(df.describe())
    print(df)
    print(df.describe().round(2))
    print(df.info())

    # Imputation by converting NA cells to median, mode, or mean.
    df = convertNAcellsToNum("bathrooms", df, "mode")
    df = convertNAcellsToNum("review_scores_rating", df, "median")
    df = convertNAcellsToNum("bedrooms", df, "mode")
    df = convertNAcellsToNum("beds", df, "mean")

    # ----------------------------
    # Outlier Treatment
    # ----------------------------

    LOWER_P = 0.01  # 1st percentile
    UPPER_P = 0.98 # 9th percentile

    lower_bound = df['price'].quantile(LOWER_P)
    upper_bound = df['price'].quantile(UPPER_P)

    print(f"Price clipping bounds: {lower_bound:.2f} to {upper_bound:.2f}")

    df['price_clipped'] = df['price'].clip(lower_bound, upper_bound)

    # Replace target with clipped version
    y = df['price_clipped']

    # ----------------------------
    # Outlier Treatment for Bedrooms, Beds, Bathrooms
    # ----------------------------

    for col in ["imp_bedrooms", "imp_beds", "imp_bathrooms"]:
        lower = df[col].quantile(0.01)  # 1st percentile
        print(f"{col} clipping lower bound: {lower:.2f}")
        df[col] = df[col].clip(lower, None)

    # ----------------------------
    # Outlier Treatment for accommodates (upper only)
    # ----------------------------

    UPPER_A = 0.99  # 99th percentile

    upper_acc = df['accommodates'].quantile(UPPER_A)
    print(f"accommodates clipping upper bound: {upper_acc:.2f}")

    df['accommodates'] = df['accommodates'].clip(None, upper_acc)

    df["city"] = df["city"].where(df["city"].isin(["LA", "SF", "NYC", "DC", "Chicago"]), "Other")

    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "property_type",
                        "room_type",
                        "bed_type",
                        "cancellation_policy",
                        "cleaning_fee",
                        "city",
                        # "first_review",
                        "host_has_profile_pic",
                        "host_identity_verified",
                        # "host_response_rate",
                        # "host_since",
                        "instant_bookable",
                        # "last_review",
                        # "neighbourhood",
                        # "zipcode",
                    ]
                ],
                columns=[
                    "property_type",
                    "room_type",
                    "bed_type",
                    "cancellation_policy",
                    "cleaning_fee",
                    "city",
                    # "first_review",
                    "host_has_profile_pic",
                    "host_identity_verified",
                    # "host_response_rate",
                    # "host_since",
                    "instant_bookable",
                    # "last_review",
                    # "neighbourhood",
                    # "zipcode",
                ],
            ).astype(int),
        ],
        axis=1,
    )  # Join dummy df with original df

    # Create bins.
    df["accommodatesBin"] = pd.cut(
        x=df["accommodates"],
        bins=[0, 2, 3, 4, 5, 6, 8, 16, 1000],
    )
    df["ratingBin"] = pd.cut(
        x=df["imp_review_scores_rating"],
        bins=[0, 60, 80, 90, 95, 100],
    )
    df["reviewsBin"] = pd.cut(
        x=df["number_of_reviews"],
        bins=[-1, 0, 10, 25, 50, 100, 300, 1000, 100000],
    )
    df["bathroomsBin"] = pd.cut(
        x=df["imp_bathrooms"],
        bins=[-1, 1, 2, 3, 4, 10, 1000],
    )
    df["bedroomsBin"] = pd.cut(
        x=df["imp_bedrooms"],
        bins=[-1, 1, 2, 3, 4, 6, 10, 1000],
    )
    df["bedsBin"] = pd.cut(
        x=df["imp_beds"],
        bins=[-1, 1, 2, 3, 4, 6, 12, 1000],
    )

    # Join dummies with original df
    df = pd.concat(
        [
            df,
            pd.get_dummies(
                df[
                    [
                        "accommodatesBin",
                        "ratingBin",
                        "reviewsBin",
                        "bathroomsBin",
                        "bedroomsBin",
                        "bedsBin",

                    ]
                ],
                columns=[
                    "accommodatesBin",
                    "ratingBin",
                    "reviewsBin",
                    "bathroomsBin",
                    "bedroomsBin",
                    "bedsBin",

                ],
            drop_first=True).astype(int),
        ],
        axis=1,
    )

    print("\nAfter imputation and dummy variables:")
    print(df)
    print(df.describe().round(2))
    print(df.info())

    x = df[
        [
            # "property_type", non-numeric, dummy
            # 'property_type_House',
            # 'property_type_Apartment',
            # 'property_type_Other',
            # 'property_type_Loft',
            # 'property_type_Condominium',
            # 'property_type_Townhouse',
            # 'property_type_Cabin',
            # 'property_type_Dorm',
            # 'property_type_Bungalow',
            # 'property_type_Villa',
            # 'property_type_In-law',
            # 'property_type_Bed & Breakfast',
            # 'property_type_Guesthouse',
            # 'property_type_Tipi',
            # 'property_type_Boutique hotel',
            # 'property_type_Camper/RV',
            # 'property_type_Boat',
            # 'property_type_Hostel',
            #  'property_type_Guest suite',
            # 'property_type_Timeshare',
            # 'property_type_Yurt',
            # 'property_type_Serviced apartment',
            # 'property_type_Tent',
            # 'property_type_Treehouse',
            # 'property_type_Castle',
            # 'property_type_Vacation home',
            # 'property_type_Earth House',
            # 'property_type_Cave',
            # 'property_type_Hut',
            # 'property_type_Island',
            # 'property_type_Chalet',
            # 'property_type_Parking Space',
            # 'property_type_Casa particular',
            # 'property_type_Train',
            # "room_type", non-numeric, dummy
            'room_type_Entire home/apt',
            'room_type_Private room',
            # 'room_type_Shared room',

            "accommodates",
            "imp_bathrooms",
            # "bed_type", non-numeric, dummy
            # 'bed_type_Real Bed',
            # 'bed_type_Couch',
            # 'bed_type_Futon',
            # 'bed_type_Airbed',
            # 'bed_type_Pull-out Sofa',

            # "cancellation_policy", non-numeric, dummy
            # 'cancellation_policy_strict',
            # 'cancellation_policy_moderate',
            # 'cancellation_policy_flexible',
            # 'cancellation_policy_super_strict_30',
            # 'cancellation_policy_super_strict_60',
            # "cleaning_fee", non-numeric, dummy
            # 'cleaning_fee_True',
            # 'cleaning_fee_False',
            # "city", #non-numeric
            # "first_review", non-numeric
            # "host_has_profile_pic", non-numeric, dummy
            # 'host_has_profile_pic_t',
            # 'host_has_profile_pic_f',
            # "host_identity_verified", non-numeric, dummy
            # 'host_identity_verified_t',
            # 'host_identity_verified_f',
            # "host_response_rate", non-numeric
            # "host_since", non-numeric
            # "instant_bookable", non-numeric, dummy
            # "instant_bookable_f",
            # "instant_bookable_t",
            # "last_review", non-numeric
            # "neighbourhood", non-numeric
            # "number_of_reviews",
            #  "m_review_scores_rating",
            # "imp_review_scores_rating",
            "imp_bedrooms",
            "imp_beds",
            # "air_conditioning",
            # "bathtub",
            # "buzzer_wireless_intercom",
            "cable_tv",
            # "carbon_monoxide_detector",
            # "crib",
            # "dishwasher",
            # "doorman",
            # "doorman_entry",
            # "elevator",
            # "family_kid_friendly",
            # "fire_extinguisher",
            # "first_aid_kit",
            # "free_parking_on_premises",
            # "gym",
            # "hangers",
            # "heating",
            # "indoor_fireplace",
            # "internet",
            # "keypad",
            # "lock_on_bedroom_door",
            # "pets_live_on_this_property",
            # "private_entrance",
            # "private_living_room",
            # "shampoo",
            # "smartlock",
            # "smoking_allowed",
            # "suitable_for_events",
            "tv",
            # dummy
            # 'accommodatesBin_(0, 2]',
            # 'accommodatesBin_(2, 3]',
            # 'accommodatesBin_(3, 4]',
            # 'accommodatesBin_(4, 5]',
            'accommodatesBin_(5, 6]',
            # 'accommodatesBin_(6, 8]',
            # 'accommodatesBin_(8, 16]',
            # 'accommodatesBin_(16, 1000]',

            # 'ratingBin_(0, 60]',
            # 'ratingBin_(60, 80]',
            # #'ratingBin_(80, 90]',
            # 'ratingBin_(90, 95]',
            # 'ratingBin_(95, 100]',

            # 'reviewsBin_(-1, 0]',
            # 'reviewsBin_(0, 10]',
            # 'reviewsBin_(10, 25]',
            # 'reviewsBin_(25, 50]',
            # #'reviewsBin_(50, 100]',
            # 'reviewsBin_(100, 300]',
            # 'reviewsBin_(300, 1000]',
            # 'reviewsBin_(1000, 100000]',

            # 'bathroomsBin_(-1, 1]',
            #'bathroomsBin_(1, 2]',
            #'bathroomsBin_(2, 3]',
            # 'bathroomsBin_(3, 4]',
            #'bathroomsBin_(4, 10]',
            # 'bathroomsBin_(10, 1000]',

            #'bedroomsBin_(-1, 1]',
            'bedroomsBin_(1, 2]',
            #'bedroomsBin_(2, 3]',
            'bedroomsBin_(3, 4]',
            'bedroomsBin_(4, 6]',
            # 'bedroomsBin_(6, 10]',
            # 'bedroomsBin_(10, 1000]',

            #'bedsBin_(-1, 1]',
            # 'bedsBin_(1, 2]',
            # 'bedsBin_(2, 3]',
            # 'bedsBin_(3, 4]',
            # 'bedsBin_(4, 6]',
            'bedsBin_(6, 12]',
            # 'bedsBin_(12, 1000]',

            # city dummies
            "city_DC",
            "city_NYC",
            "city_SF",
            #"city_LA"
            #"city_Chicago",
            #"city_Other",

        ]
    ]

    # Target variable
    y = df['price_clipped']

    # used for feature selection
    # #  f_regression returns F statistic for each feature.
    # ffs = f_regression(x, y)
    #
    # featuresDf = pd.DataFrame()
    # for i in range(0, len(x.columns)):
    #     featuresDf = featuresDf._append({"feature": x.columns[i],
    #                                      "ffs": ffs[0][i]}, ignore_index=True)
    # featuresDf = featuresDf.sort_values(by=['ffs'])
    # print(featuresDf)
    # print("\nTop 10 Features Based on F-Regression:")
    # print(featuresDf.tail(30))


    x = sm.add_constant(x)

    print("\nCross fold validation:")
    cross_fold_validation(x, y, 4)


    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85)

    # used for checking
    from sklearn.preprocessing import StandardScaler

    # from sklearn.preprocessing import RobustScaler
    # sc_x    = RobustScaler()
    # X_Scale = sc_x.fit_transform(x)
    #
    #
    # X_train_scaled  = sc_x.fit_transform(x_train) # Fit and transform X.
    # X_test_scaled   = sc_x.transform(x_test)      # Transform X.
    #
    # model = sm.OLS(y_train, X_train_scaled).fit()
    # predictions = model.predict(X_test_scaled)

    model = sm.OLS(y_train, x_train).fit()
    predictions = model.predict(x_test)

    print(model.summary(title="Model C: Price Prediction"))
    print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    draw_validation_plots(
        title="Model C: Price",
        bins=8,
        y_test=y_test,
        predictions=predictions,
    )



def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    model_c()

if __name__ == "__main__":
        main()