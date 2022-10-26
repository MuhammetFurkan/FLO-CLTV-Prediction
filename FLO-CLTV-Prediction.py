# Bussines Problem

""""
FLO would like to set a roadmap for sales and marketing activities.
In order for the company to make a medium-long-term plan,
The company needs to estimate the potential value that existing customers will provide to the company in the future.
"""

# Features:

# Total Features : 12
# Total Row : 19.945
# CSV File Size : 2.7 MB

""""
- master_id : Unique Customer Number
- order_channel : Which channel of the shopping platform is used (Android, IOS, Desktop, Mobile)
- last_order_channel : The channel where the most recent purchase was made
- first_order_date : Date of the customer's first purchase
- last_order_channel : Customer's previous shopping history
- last_order_date_offline : The date of the last purchase made by the customer on the offline platform
- order_num_total_ever_online : Total number of purchases made by the customer on the online platform
- order_num_total_ever_offline : Total number of purchases made by the customer on the offline platform
- customer_value_total_ever_offline : Total fees paid for the customer's offline purchases
- customer_value_total_ever_online :  Total fees paid for the customer's online purchases
- interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
"""




############################################
# Importing Library and Functions
############################################


import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


df_ = pd.read_csv('flo_data_20k.csv')
df = df_.copy()

# Check DataFrame - Function

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

# Outlier Analysis

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

# Suppressing outliers
columns = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]

for i in columns:
    replace_with_thresholds(df,i)

# Omnichannel means that customers shop from both online and offline platforms.
# Create new variables for each customer's total purchases and spending.

df["total_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

#  Change the type of date expressing variables to date.

df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

############################################
# Creating the CLTV Data Structure
############################################

# Take 2 days after the date of the last purchase in the data set as the date of analysis.

df["last_order_date"].max()
analysis_date = dt.datetime(2021, 6, 1)

# Create a new CLTV dataframe with customer_id, recency_cltv_weekly, T_weekly, frequency and monetary_cltv_avg.

# recency: Time since last purchase. Weekly (user specific)
# T_weekly: Age of the customer. Weekly (how long before the date of analysis was the first purchase)
# frequency: total number of repeat purchases (frequency>1)
# monetary: average earnings per purchase


cltv_df = pd.DataFrame()
cltv_df["customer_id"] = df["master_id"]
cltv_df["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) / 7
cltv_df["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64")) / 7
cltv_df["frequency"] = df["total_purchases"]
cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["total_purchases"]

cltv_df.head()


############################################
# BG/NBD, Gamma-Gamma Model
############################################

# Installation of BG/NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df['frequency'],
        cltv_df['recency_cltv_weekly'],
        cltv_df['T_weekly'])

# Estimate expected purchases from customers in 3 months and add exp_sales_3_month to the CLTV data frame.

cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Estimate expected purchases from customers in 6 months and add exp_sales_6_month to the CLTV data frame.

cltv_df["exp_sales_6_month"] = bgf.predict(4*6,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'])

# Fit the Gamma-Gamma model. Estimate the average value of the customers and add it to the CLTV dataframe as exp_average_value.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])


# Calculate 6 months CLTV and add it to the dataframe with the name CLTV.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,
                                   freq="W",
                                   discount_rate=0.01)
cltv_df["cltv"] = cltv

# Top 20 people with the highest CLTV.
cltv_df.sort_values("cltv",ascending=False)[:20]

############################################
# Creating Segments by CLTV
############################################

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

############################################
# Functionalize the whole process
############################################
def create_cltv_df(dataframe):

    # Preparing the Data
    columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline","customer_value_total_ever_online"]
    for col in columns:
        replace_with_thresholds(dataframe, col)

    dataframe["order_num_total"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_total"] = dataframe["customer_value_total_ever_offline"] + dataframe["customer_value_total_ever_online"]
    dataframe = dataframe[~(dataframe["customer_value_total"] == 0) | (dataframe["order_num_total"] == 0)]
    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    # Creation of CLTV data structure
    dataframe["last_order_date"].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    cltv_df = pd.DataFrame()
    cltv_df["customer_id"] = dataframe["master_id"]
    cltv_df["recency_cltv_weekly"] = ((dataframe["last_order_date"] - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["T_weekly"] = ((analysis_date - dataframe["first_order_date"]).astype('timedelta64[D]')) / 7
    cltv_df["frequency"] = dataframe["order_num_total"]
    cltv_df["monetary_cltv_avg"] = dataframe["customer_value_total"] / dataframe["order_num_total"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

    # BG-NBD Model
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])
    cltv_df["exp_sales_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])
    cltv_df["exp_sales_6_month"] = bgf.predict(4 * 6,
                                               cltv_df['frequency'],
                                               cltv_df['recency_cltv_weekly'],
                                               cltv_df['T_weekly'])

    # Gamma-Gamma Model
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                           cltv_df['monetary_cltv_avg'])

    # Cltv prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_avg'],
                                       time=6,
                                       freq="W",
                                       discount_rate=0.01)
    cltv_df["cltv"] = cltv

    # CLTV segmentation
    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])

    return cltv_df

cltv_df = create_cltv_df(df)
cltv_df.head(10)