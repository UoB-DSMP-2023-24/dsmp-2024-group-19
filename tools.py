import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

data_folder = "Data"
lob_subfolder = "LOBs"
tapes_subfolder = "Tapes"

def get_LOBs(n: int = 0, min_n: int = 0) -> list[pd.DataFrame]:
    """ 
    Retrieves limit order book (LOB) data.
    
    Args:
        n (int): Number of LOBs to retrieve. Default is 0.
        min_n (int): Minimum index of LOBs to retrieve. Default is 0.
        
    Returns:
        list[pd.DataFrame]: List of DataFrames containing the LOB data.
    """
    assert n >= min_n
    assert min_n >= 0
    assert n < 125

    LOB_filenames = os.listdir(os.path.join(data_folder, lob_subfolder))

    raw_LOBs = []

    for filename in LOB_filenames[min_n:n+1]:
        print(f"Opening {filename}")
        if filename[:10] != "UoB_Set01_":
            print("Invalid Filename:", filename)
        else:
            date = filename[10:20]
            
            with open(os.path.join(data_folder, lob_subfolder, filename), 'r') as f:
                lob_raw = f.readlines()

            lob_list = []
            for row in lob_raw:
                parsed_row = ast.literal_eval(row.replace("Exch0", "'Exch0'"))
                
                high_bid = None
                low_ask = None
                
                if len(parsed_row[2][0][1]) > 0:
                    high_bid = -np.inf
                    for bid, vol in parsed_row[2][0][1]:
                        if bid > high_bid:
                            high_bid = bid

                if len(parsed_row[2][1][1]) > 0:
                    low_ask = np.inf
                    for ask, vol in parsed_row[2][1][1]:
                        if ask < low_ask:
                            low_ask = ask

                lob_list.append(parsed_row + [high_bid, low_ask])

            df = pd.DataFrame(lob_list)

            column_mapping = {
                0 : "Seconds",
                1 : "Exchange",
                2 : "LOB",
                3 : "high_bid",
                4 : "low_ask"
            }

            df.rename(columns=column_mapping, inplace=True)

            df['Date'] = pd.to_datetime(date)
            df['Seconds'] = pd.to_timedelta(df["Seconds"], unit="s")
            df['combined_time'] = df['Date'] + df['Seconds'] + pd.Timedelta(hours=8)
            df.index = df['combined_time']
            df = df.drop(['Date', 'Seconds', 'combined_time'], axis=1)

            df["mid_price"] = (df["high_bid"] + df["low_ask"]) / 2

            raw_LOBs.append(df)

            if len(raw_LOBs) >= n:
                return raw_LOBs

    return raw_LOBs

def get_Tapes(n: int = 0, min_n: int = 0) -> list[pd.DataFrame]:
    """
    Retrieves a specified number of Tape dataframes.

    Args:
        n (int, optional): Number of Tape dataframes to retrieve. Defaults to 0.
        min_n (int, optional): Minimum index from which to retrieve Tape dataframes. Defaults to 0.

    Returns:
        list[pd.DataFrame]: A list of Tape dataframes.

    Raises:
        AssertionError: If n is less than min_n, min_n is less than 0, or n is greater than or equal to 125.
    """
    assert n >= min_n
    assert min_n >= 0
    assert n < 125

    Tapes_filenames = os.listdir(os.path.join(data_folder, tapes_subfolder))

    raw_tapes = []
    for filename in Tapes_filenames[min_n:n+1]:
        print(f"Opening {filename}")
        if filename[:10] != "UoB_Set01_":
            print("Invalid Filename:", filename)
        else:
            date = filename[10:20]
            
            df = pd.read_csv(os.path.join(data_folder, tapes_subfolder, filename), header=None)
            column_mapping = {
                0: "Seconds",
                1: "Price",
                2: "Volume",
            }

            df.rename(columns=column_mapping, inplace=True)

            df['Date'] = pd.to_datetime(date)
            df['Seconds'] = pd.to_timedelta(df["Seconds"], unit="s")
            df['combined_time'] = df['Date'] + df['Seconds'] + pd.Timedelta(hours=8)

            df.index = df['combined_time']

            df = df.drop(['Date', 'Seconds', 'combined_time'], axis=1)
            
            raw_tapes.append(df)

            if len(raw_tapes) >= n:
                return raw_tapes

    return raw_tapes

def list_diff(list1, list2):
    """
    Returns the elements in list1 that are not present in list2.

    Parameters:
    - list1 (list): The first list.
    - list2 (list): The second list.

    Returns:
    - list: A list containing elements from list1 that are not present in list2.
    """
    return [x for x in list1 if x not in list2]

def clean_lob(df):
    """
    Cleans the Limit Order Book (LOB) data.

    Parameters:
    - df (DataFrame): DataFrame containing the LOB data.

    Returns:
    - DataFrame: Cleaned DataFrame.
    """
    b_val = 1
    c = 0

    for i, row in df.iterrows():
        print(i, end = "\r")
        if c == 0:
            pass
        else:
            bid = row["LOB"][0][1]
            ask = row["LOB"][1][1]

            df.at[i, "Incoming bid"] = str(list_diff(bid, prev_bid))
            df.at[i, "Incoming ask"] = str(list_diff(ask, prev_ask))

            df.at[i, "Outgoing bid"] = str(list_diff(prev_bid, bid))
            df.at[i, "Outgoing ask"] = str(list_diff(prev_ask, ask))

            df.at[i, "Number_bids"] = len(row["LOB"][0][1])
            df.at[i, "Number_asks"] = len(row["LOB"][1][1])

            if ~np.isnan(row["mid_price"]):
                asks = row["LOB"][1][1]
                bids = row["LOB"][0][1]
                mid = row["mid_price"]

                alpha = 0
                for a, num in asks:
                    assert a > mid
                    alpha += num/((a - mid) + b_val)

                beta = 0
                for b, num in bids:
                    assert mid > b
                    beta += num /((mid - b) + b_val)

                df.at[i, "alpha"] = alpha
                df.at[i, "beta"] = beta
            
        c += 1
        prev_bid = row["LOB"][0][1]
        prev_ask = row["LOB"][1][1]

    return df

def resample_LOB(df):
    """
    Resamples the cleaned LOB data into the 1hz domain.

    Parameters:
    - df (DataFrame): DataFrame containing the cleaned LOB data.

    Returns:
    - DataFrame: Resampled DataFrame.
    """
    df_bids_asks = df[["Incoming bid", "Incoming ask", "Outgoing bid", "Outgoing ask"]]
    df_bids_asks = df_bids_asks.applymap(lambda x: x[1:-1].replace(",","") if isinstance(x, str) else "")
    df_bids_asks = df_bids_asks.resample("1s").sum()

    df_AB = df[["alpha", "beta"]]
    df_AB = df_AB.resample("1s").mean() # taking the average alpha and beta value over the second

    df_prices = df[["LOB","mid_price", "low_ask", "high_bid"]]
    df_prices = df_prices.resample("1s").last()
    #df_prices["LOB"] = df_prices["LOB"].fillna("[]").apply(ast.literal_eval)

    return pd.concat([df_bids_asks, df_AB, df_prices], axis = 1)

def resample_Tapes(df):
    """
    Resamples the tapes data into the 1hz domain.

    Parameters:
    - df (DataFrame): DataFrame containing the tapes data.

    Returns:
    - DataFrame: Resampled DataFrame.
    """
    df["Price x Volume"] = df["Price"] * df["Volume"]
    resampled_df = df.resample("1s").sum()
    resampled_df["Tapes Price"] = resampled_df["Price x Volume"] / resampled_df["Volume"]
    resampled_df["Last Tapes Price"] = resampled_df["Tapes Price"].ffill()
    resampled_df.drop(["Price","Price x Volume"], axis = 1, inplace = True)

    return resampled_df

def read_merged_data(n: int = 0, min_n: int = 0) -> list[pd.DataFrame]:
    """
    Reads and merges the resampled LOB and tapes data.

    Parameters:
    - n (int): Number of files to read and merge.
    - min_n (int): Minimum file number.

    Returns:
    - list: List of merged DataFrames.
    """
    assert n >= min_n
    assert min_n >= 0
    assert n < 125

    output = []

    for i in range(min_n, n + 1):
        LOB = pd.read_csv(f"Processed_Data/Clean_LOB/Clean_LOB_{i}.csv", index_col=0, parse_dates=True, engine="pyarrow")
        Tapes = get_Tapes(i,i)[0]

        LOB_resample = resample_LOB(LOB)
        Tapes_resample = resample_Tapes(Tapes)

        output.append(pd.concat([LOB_resample, Tapes_resample], axis = 1))

    return output

def move_to_parent_dir():
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    os.chdir(parent_directory)
    print("Working directory:", os.getcwd())