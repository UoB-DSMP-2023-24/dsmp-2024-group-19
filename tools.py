import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

data_folder = "Data"
lob_subfolder = "LOBs"
tapes_subfolder = "Tapes"

def get_LOBs(n = 125) -> list[pd.DataFrame]:
    """ prototype: use with caution"""
    LOB_filenames = os.listdir(data_folder + '\\' + lob_subfolder)

    raw_LOBs = []

    for filename in LOB_filenames:
        print(f"Opening {filename}")
        if filename[:10] != "UoB_Set01_":
            print("Invalid Filename:", filename)
        else:
            date = filename[10:20]
            
            with open(data_folder + "\\" + lob_subfolder + "\\" + filename, 'r') as f:
                lob_raw = f.readlines()

            lob_list = [ast.literal_eval(a.replace("Exch0", "'Exch0'")) for a in lob_raw]

            df = pd.DataFrame(lob_list)

            column_mapping = {
                0 : "Seconds",
                1 : "Exchange",
                2 : "LOB",
            }

            df.rename(columns = column_mapping, inplace = True)

            df['Date'] = pd.to_datetime(date)

            df['Seconds'] = pd.to_timedelta(df["Seconds"], unit = "s")

            df['combined_time'] = df['Date'] + df['Seconds'] + pd.Timedelta(hours=8)

            df.index = df['combined_time']

            df = df.drop(['Date', 'Seconds', 'combined_time'], axis=1)

            raw_LOBs.append(df)

            if len(raw_LOBs) >= n:
                return raw_LOBs

    return raw_LOBs

def get_Tapes(n = 125) -> list[pd.DataFrame]:
    
    Tapes_filenames = os.listdir(data_folder + '\\' + tapes_subfolder)

    raw_tapes = []
    for filename in Tapes_filenames:
        print(f"Opening {filename}")
        if filename[:10] != "UoB_Set01_":
            print("Invalid Filename:", filename)
        else:
            date = filename[10:20]
            
            df = pd.read_csv(data_folder + "\\" + tapes_subfolder + "\\" + filename, header = None)

            column_mapping = {
                0 : "Seconds",
                1 : "Price",
                2 : "Volume",
            }

            df.rename(columns = column_mapping, inplace = True)

            df['Date'] = pd.to_datetime(date)

            df['Seconds'] = pd.to_timedelta(df["Seconds"], unit = "s")

            df['combined_time'] = df['Date'] + df['Seconds'] + pd.Timedelta(hours=8)

            df.index = df['combined_time']

            df = df.drop(['Date', 'Seconds', 'combined_time'], axis=1)
            
            raw_tapes.append(df)

            if len(raw_tapes) >= n:
                return raw_tapes

    return raw_tapes