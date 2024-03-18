import os
from scipy.sparse import load_npz
import numpy as np
from datetime import datetime, timedelta
import time
import numpy as np
from numba import njit, prange

def get_dates():
    # Define the start and end dates
    start_date = datetime(2025, 1, 2)
    end_date = datetime(2025, 7, 1)

    # Generate a list of dates excluding weekends
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Sunday = 6
            date_list.append(current_date)
        current_date += timedelta(days=1)

    date_list = [date.strftime("%Y-%m-%d") for date in date_list]
    holidays = ["2025-04-18","2025-04-21", "2025-05-05", "2025-05-26"]
    for h in holidays:
        date_list.remove(h)

    #print(len(date_list), "days between", date_list[0], date_list[-1])
    return date_list

def readc_day(day: str):
    folder = "CSR_Data"
    lob_data = load_npz(os.path.join(folder,"CSR_LOB_"+day+".npz")).toarray()
    lob_times = np.load(os.path.join(folder, "TIM_LOB_"+day+".npy"))
    tapes = np.load(os.path.join(folder, "TAP_"+day+".npy"))
    return lob_data, lob_times, tapes

def get_data(min_n = 0, max_n = 125):
    s = time.time()
    date_list = get_dates()
    all_data = []

    for day in date_list[min_n:max_n]:
        data = readc_day(day)
        all_data.append(data)

    print("Time taken to reach each day:", (time.time() - s) / (max_n - min_n))

    return all_data

@njit
def skewness_kurtosis(data: np.array):
    # Calculate sum
    sum_ = np.sum(data)

    # Calculate mean
    mean = np.mean(data)
    
    # Calculate standard deviation
    std_dev = np.std(data)
    
    # Calculate skewness
    skewness = np.mean((data - mean) ** 3) / (std_dev ** 3)
    
    # Calculate kurtosis
    kurtosis = np.mean((data - mean) ** 4) / (std_dev ** 4) - 3
    
    return sum_, mean, std_dev, skewness, kurtosis

@njit(parallel=True)
def get_features(lob_data: np.array, 
                 lob_times: np.array, 
                 tapes: np.array, 
                 time_step_s: int, 
                 ab_weight = 1, 
                 median = False, 
                 cas_cbs_window = 800):
    """
    Calculate features from LOB and Tapes data.

    Parameters:
    -----------
    lob_data : np.array
        Array containing the limit order book (LOB) data.
    lob_times : np.array
        Array containing timestamps for the LOB data.
    tapes : np.array
        Array containing Tapes data.
    time_step_s : int
        Time step in seconds for calculating features.
    ab_weight : float, optional
        Weight parameter for alpha and beta calculations, by default 1.
    median : bool, optional
        Whether to calculate features using median instead of mean, by default False.
    cas_cbs_window : int, optional
        Size of the window for calculating CAS and CBS, by default 800.

    Returns:
    --------
    tuple
        A tuple containing:
        - feat_arr: np.array
            Array containing feature values.
        - time_arr: np.array
            Array containing timestamps.
        - features: list
            List of feature names.
    """
    
    n_rows = int((8.5 * 60 * 60) / time_step_s)                         # define number of rows of output array
    features = ["MP","HIBID","LOASK","AP","WBP","WAP",                  # define features
                "TCBS","TCAS","AWS","VOL","GAP","SPREAD",
                "ALPHA", "BETA", "ZETA", "ENDT"]
    n_features = len(features)                                          # define number of features

    feat_arr = np.zeros((n_rows, n_features), dtype=np.float64)         # array to hold feature values
    
    LA_HB_a_b = np.zeros((lob_data.shape[0]+1, 4), dtype = np.float64)  # array holding the LOASK, HIBID,
                                                                        # alpha, beta, values 

    for i in prange(lob_data.shape[0]):                                 # iterates over the LOB to fill
        row = lob_data[i]                                               # LA_HB_a_b values
        
        neg_ind = np.where(row < 0)[0]                                  # locate bid and ask prices (indicies)
        pos_ind = np.where(row > 0)[0]
        
        if len(neg_ind) == 0:                                           # assign HIBID, np.nan if no values
            LA_HB_a_b[i][1] = np.nan
        else:
            LA_HB_a_b[i][1] = max(neg_ind) + 1 

        if len(pos_ind) == 0:                                           # assign HIBID, np.nan if no values
            LA_HB_a_b[i][0] = np.nan
        else:
            LA_HB_a_b[i][0] = min(pos_ind) + 1

        mid_price = (LA_HB_a_b[i][0] + LA_HB_a_b[i][1]) / 2             # calculate mid_price for alpha/beta calculations

        if np.isnan(mid_price):
            alpha = np.nan
            beta = np.nan
        else:                                                           # calculate alpha/beta using ab_weight var
            beta = 0
            for ind in neg_ind:
                beta += (-1 * row[ind]) / ((mid_price - (ind + 1)) + ab_weight)
    
            alpha = 0
            for ind in pos_ind:
                alpha += row[ind] / (((ind + 1) - mid_price) + ab_weight)
                

        LA_HB_a_b[i][2] = alpha
        LA_HB_a_b[i][3] = beta
        
    max_lob = lob_data.shape[0] - 1                                      # define max indicies for lob
    max_tapes = tapes.shape[0] - 1                                       # define max indicies for tapes
    
    start_time = 0                                                       # define start time
    lob_start = 0                                                        # define start index for lob
    tapes_start = 0                                                      # define start index for tapes
    
    cas = np.zeros(800, dtype = np.int16)                                # define an array to hold CAS values
    cbs = np.zeros(800, dtype = np.int16)                                # define an array to hold CBS values
    for row_i in range(n_rows):
        end_time = start_time + time_step_s                              # move to next time step
        lob_end = lob_start
        tapes_end = tapes_start

        # get lob end index
        while lob_times[lob_end] < end_time and lob_end < max_lob:       # move lob indicies to end time
            lob_end += 1
        
        # get tapes end index
        while tapes[tapes_end][0] < end_time and tapes_end < max_tapes:  # move tapes indicies to end time
            tapes_end += 1

        # feature calculations
        if tapes_start == tapes_end:                                     # if there is no tapes data
            AP = np.nan                                                  # set tapes features to np.nan
            VOL = np.nan
        else:
            tapes_slice = tapes[tapes_start:tapes_end]                   # extract tapes slice, calculate AP, VOL
            AP = 0
            for row in tapes_slice:
                AP += row[1] * row[2]
            VOL = np.sum(tapes_slice[:,2])
            AP = AP / VOL

        if lob_start == lob_end:                                         # if there is no LOB data
            MP = np.nan                                                  # set lob features to np.nan
            HIBID = np.nan
            LOASK = np.nan
            SPREAD = np.nan
            TCBS = np.nan
            TCAS = np.nan
            WBP = np.nan
            WAP = np.nan
            AWS = np.nan
            ALPHA = np.nan
            BETA = np.nan
            ZETA = np.nan  
        else:
            lob_slice = lob_data[lob_start:lob_end]                       # extract slices of data 
            LA_HB_a_b_slice = LA_HB_a_b[lob_start:lob_end]                

            # midprice_calcs, alpha, beta
            if median:                                                    # calculate price features
                HIBID = np.median(LA_HB_a_b_slice[:,1])                   # using median if set to true
                LOASK = np.median(LA_HB_a_b_slice[:,0])
                ALPHA = np.median(LA_HB_a_b_slice[:,2])
                BETA = np.median(LA_HB_a_b_slice[:,3])
            else:
                HIBID = np.nanmean(LA_HB_a_b_slice[:,1])
                LOASK = np.nanmean(LA_HB_a_b_slice[:,0])
                ALPHA = np.nanmean(LA_HB_a_b_slice[:,2])
                BETA = np.nanmean(LA_HB_a_b_slice[:,3])

            MP = (HIBID + LOASK) / 2
            SPREAD = LOASK - HIBID
            ZETA = BETA - ALPHA

            if HIBID >= LOASK:
                print("WARNING: HIBID >= LOASK")

            # consolidated calcs
            cas[:] = 0                                                      # reset cas, cbs arrays for new data
            cbs[:] = 0 
            idx_MP = int(MP - 1)                                            # calculate index for mid_price
            for ci in prange(800):
                # can optimise with LOASK AND HIBID here
                p = ci + 1

                if p <= LOASK + 100 and p >= idx_MP - cas_cbs_window:       # only calculate cbs between window left of MP
                    cbs_vec = lob_slice[:,ci].copy() * -1                   # and less than LOASK + 100 for efficiency
                    cbs_vec[cbs_vec <= 0] = 0                               # idk if this breaks things for efficiency ?:
                    cbs[ci] = np.sum(np.abs(np.diff(cbs_vec))) + cbs_vec[0]

                if p >= HIBID - 100 and p <= idx_MP + cas_cbs_window:       # only calculate cas between window right of MP
                    cas_vec = lob_slice[:,ci].copy()                        # and greater than HIBID - 100 for efficiency
                    cas_vec[cas_vec <= 0] = 0                               # idk if this breaks things for efficiency ?:
                    cas[ci] = np.sum(np.abs(np.diff(cas_vec))) + cas_vec[0]

            TCBS = np.sum(cbs)                                              # Total CBS
            TCAS = np.sum(cas)                                              # Total CAS

            if TCBS == 0:                                                   # Calculate WBP, np.nan if no activity
                WBP = np.nan
            else:
                WBP = 0
                for ci in prange(800):
                    WBP += (ci + 1) * (cbs[ci] / TCBS)

            if TCAS == 0:                                                   # Calculate WAP, np.nan if no activity
                WAP = np.nan
            else:
                WAP = 0
                for ci in prange(800):
                    WAP += (ci + 1) * (cas[ci] / TCAS)

            AWS = WAP - WBP                                                 # Activity weighted spread calc

        # feature setting
        feat_arr[row_i][features.index("AP")] = AP                          # set the values to the feat_arr
        feat_arr[row_i][features.index("VOL")] = VOL
        feat_arr[row_i][features.index("MP")] = MP
        feat_arr[row_i][features.index("HIBID")] = HIBID
        feat_arr[row_i][features.index("LOASK")] = LOASK
        feat_arr[row_i][features.index("SPREAD")] = SPREAD
        feat_arr[row_i][features.index("TCAS")] = TCAS
        feat_arr[row_i][features.index("TCBS")] = TCBS
        feat_arr[row_i][features.index("WAP")] = WAP
        feat_arr[row_i][features.index("WBP")] = WBP
        feat_arr[row_i][features.index("AWS")] = AWS
        feat_arr[row_i][features.index("ALPHA")] = ALPHA
        feat_arr[row_i][features.index("BETA")] = BETA
        feat_arr[row_i][features.index("ZETA")] = ZETA
        feat_arr[row_i][features.index("GAP")] = MP - AP
        feat_arr[row_i][features.index("ENDT")] = end_time


        # adjust start times
        start_time = end_time                                                # Set the next start times and 
        lob_start = lob_end                                                  # indicies to the last end times / indicies
        tapes_start = tapes_end

    # =================================================
    # calculate daily features
    daily_arr = {}
    # volume features
    vt, vm, vs, vw, vk = skewness_kurtosis(tapes[:,2])

    daily_arr["VOL_SUM"] = vt
    daily_arr["VOL_MEAN"] = vm
    daily_arr["VOL_STD"] = vs
    daily_arr["VOL_SKEW"] = vw
    daily_arr["VOL_KURT"] = vk

    # price x volume features
    price_x_vol = tapes[:,1] * tapes[:,2]
    vt, vm, vs, vw, vk = skewness_kurtosis(price_x_vol)

    daily_arr["VOL_SUM"] = vt
    #daily_arr["PVOL_MEAN"] = vm
    daily_arr["PVOL_STD"] = vs
    daily_arr["PVOL_SKEW"] = vw
    daily_arr["PVOL_KURT"] = vk

    # price features
    max_price = np.max(tapes[:,1])
    min_price = np.min(tapes[:,1])

    all_prices = np.zeros(int(daily_arr["VOL_SUM"]), dtype = np.int16)
    counter = 0
    for n, p in zip(tapes[:,1], tapes[:,2]):
        for _ in range(n):
            all_prices[counter] = p
            counter += 1

    vt, vm, vs, vw, vk = skewness_kurtosis(all_prices)

    daily_arr["PRICE_DIFF"] = max_price - min_price
    daily_arr["PRICE_STD"] = vs
    daily_arr["PRICE_SKEW"] = vw
    daily_arr["PRICE_KURT"] = vk

    return feat_arr, features, daily_arr

def load_data(time_step_s = 60, 
                 ab_weight = 1, 
                 median = False, 
                 cas_cbs_window = 800):
    
    dates = get_dates()
    output_data = []
    load_times = []
    for i, date in enumerate(dates):
        if i != 0:
            load_times.append(e-s)
            print(f"On day {i+1}/125, estimated time left = {np.median(load_times) * (124 - i):.1f}s\t\t", end = "\r")
        else:
            print("Compiling code...")

        s = time.time()
        lob, lob_time, tapes = readc_day(date) # variable names are reused so only required data is needed
        feat_arr, features, daily_arr = get_features(lob, lob_time, tapes, time_step_s, ab_weight, median, cas_cbs_window)
        output_data.append((feat_arr, daily_arr))
        e = time.time()

    return output_data, features

