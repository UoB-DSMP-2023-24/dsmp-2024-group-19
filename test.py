@njit(parallel=True)
def get_features(lob_data: np.array, lob_times: np.array, tapes: np.array, time_step_s: int, ab_weight = 1):
    n_rows = int((8.5 * 60 * 60) / time_step_s)
    print("Max time: ", (8.5*60*60))
    features = ["MP","VOL","HIBID","LOASK","SPREAD","AP",
                "TCBS","TCAS","WBP","WAP","AWS",
                "ALPHA", "BETA"]
    n_features = len(features)

    feat_arr = np.zeros((n_rows, n_features), dtype=np.float64)
    time_arr = np.arange(0,(8.5*60*60), time_step_s) + time_step_s

    LA_HB_a_b = np.zeros((lob_data.shape[0]+1, 4), dtype = np.float64)

    for i in prange(lob_data.shape[0]):
        row = lob_data[i]
        
        neg_ind = np.where(row < 0)[0]
        pos_ind = np.where(row > 0)[0]
        
        if len(neg_ind) == 0:
            LA_HB_a_b[i][1] = np.nan
        else:
            LA_HB_a_b[i][1] = max(neg_ind) + 1 # high_bid

        if len(pos_ind) == 0:
            LA_HB_a_b[i][0] = np.nan
        else:
            LA_HB_a_b[i][0] = min(pos_ind) + 1

        mid_price = (LA_HB_a_b[i][0] + LA_HB_a_b[i][1]) / 2

        if np.isnan(mid_price):
            alpha = np.nan
            beta = np.nan
        else:
            
            beta = 0
            for ind in neg_ind: # bids
                beta += (-1 * row[ind]) / ((mid_price - (ind + 1)) + ab_weight)
    
            alpha = 0
            for ind in pos_ind:
                alpha = row[ind] / (((ind + 1) - mid_price) + ab_weight)

        LA_HB_a_b[i][2] = alpha
        LA_HB_a_b[i][3] = beta
        
    max_lob = lob_data.shape[0] - 1
    max_tapes = tapes.shape[0] - 1
    
    start_time = 0
    lob_start = 0
    tapes_start = 0

    
    cas = np.zeros(800, dtype = np.int16)
    cbs = np.zeros(800, dtype = np.int16)
    for row_i in range(n_rows):
        end_time = start_time + time_step_s
        lob_end = lob_start
        tapes_end = tapes_start
        #print(start_time, end_time)

        # get lob slice
        
        while lob_times[lob_end] < end_time and lob_end < max_lob:
            lob_end += 1
        
        # get tapes slice
        while tapes[tapes_end][0] < end_time and tapes_end < max_tapes:
            tapes_end += 1

        # feature calculations
        if tapes_start == tapes_end: # if there is no tapes data
            AP = np.nan
            VOL = np.nan
        else:
            tapes_slice = tapes[tapes_start:tapes_end]
            AP = np.mean(tapes_slice[:,1])
            VOL = np.sum(tapes_slice[:,2])

        if lob_start == lob_end: # if there is no LOB data
            MP = np.nan
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
            
        else:
            lob_slice = lob_data[lob_start:lob_end]
            LA_HB_a_b_slice = LA_HB_a_b[lob_start:lob_end]

            # midprice_calcs
            HIBID = np.median(LA_HB_a_b_slice[:,1])
            LOASK = np.median(LA_HB_a_b_slice[:,0])
            MP = (HIBID + LOASK) / 2
            SPREAD = LOASK - HIBID

            if HIBID >= LOASK:
                print("WARNING: HIBID >= LOASK")

            # consolidated calcs
            cas = cas * 0
            cbs = cbs * 0
            for ci in prange(800):
                # can optimise with LOASK AND HIBID here
                if (ci + 1) <= LOASK + 100:
                    cbs_vec = lob_slice[:,ci].copy() * -1
                    cbs_vec[cbs_vec <= 0] = 0
                    cbs[ci] = np.sum(np.abs(np.diff(cbs_vec))) + cbs_vec[0]

                if (ci + 1) >= HIBID - 100:
                    cas_vec = lob_slice[:,ci].copy()
                    cas_vec[cas_vec <= 0] = 0
                    cas[ci] = np.sum(np.abs(np.diff(cas_vec))) + cas_vec[0]

            TCBS = np.sum(cbs)
            TCAS = np.sum(cas)

            if TCBS == 0:
                WBP = np.nan
            else:
                WBP = 0
                for ci in prange(800):
                    WBP += (ci + 1) * (cbs[ci] / TCBS)

            if TCAS == 0:
                WAP = np.nan
            else:
                WAP = 0
                for ci in prange(800):
                    WAP += (ci + 1) * (cas[ci] / TCAS)

            AWS = WAP - WBP
        

        # feature setting

        feat_arr[row_i][features.index("AP")] = AP
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
        

        # adjust start times
        start_time = end_time
        lob_start = lob_end
        tapes_start = tapes_end

    return feat_arr, time_arr, features


feat_arr = get_features(data[0][0],data[0][1],data[0][2],60)
s = time.time()
feat_arr, time_arr, feats = get_features(data[0][0],data[0][1],data[0][2], 60)
print(time.time() - s)
feat_arr