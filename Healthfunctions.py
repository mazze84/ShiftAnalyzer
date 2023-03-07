from datetime import datetime
import numpy as np
import pandas as pd

def getMinMaxHeartRate(heartrateMap):
    min_heartrate = 1000
    max_heartrate = 0
    avg = 0
    cnt = 0
    for time, heartrate in heartrateMap.items():
        avg += heartrate
        cnt += 1
        if heartrate > max_heartrate:
            max_heartrate = heartrate
        if heartrate < min_heartrate:
            min_heartrate = heartrate
    avg = calc_avg_heartrate(avg, cnt)
    return min_heartrate,max_heartrate,avg

def calc_avg_heartrate(heartrate, cnt):
    avg = 0
    if cnt > 0:
        heartrate /= cnt
        avg = round(heartrate, 0)
    return avg

def calc_timestamps(start_time, end_time):
    start = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
    end = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

    return end-start

def calc_shift_len(heartrateMap, active_heartrate=100):
    shifts = {}
    shift_cnt = 0

    avg_heartrate = 0
    heartrate_cnt = 0

    start_time = list(heartrateMap.keys())[0]
    last_heartrate = list(heartrateMap.values())[0]

    for time, heartrate in heartrateMap.items():

        # continue bench or active shift
        if last_heartrate < active_heartrate and heartrate < active_heartrate or last_heartrate > active_heartrate and heartrate > active_heartrate:
            avg_heartrate += heartrate
            heartrate_cnt += 1

        # start active shift or bench
        elif last_heartrate < active_heartrate and heartrate > active_heartrate or last_heartrate > active_heartrate and heartrate < active_heartrate:
            # save shift
            duration_last_shift = calc_timestamps(start_time, time)
            shifts[shift_cnt] = duration_last_shift, calc_avg_heartrate(avg_heartrate, heartrate_cnt)

            # reset values
            shift_cnt += 1
            start_time = time
            avg_heartrate = heartrate
            heartrate_cnt = 1
    duration_last_shift = calc_timestamps(start_time, list(heartrateMap.keys())[len(list(heartrateMap.keys()))-1])
    shifts[shift_cnt] = duration_last_shift, calc_avg_heartrate(avg_heartrate, heartrate_cnt)
    return shifts


def moving_average_exponential(list, x):
    numbers_series = pd.Series(list)

    # Get the moving averages of series
    # of observations till the current time
    moving_averages = round(numbers_series.ewm(alpha=0.5, adjust=False).mean(), 0)

    # Convert pandas series back to list
    return moving_averages.tolist()

def moving_average_cumulative(arr, window_size):
    window_size = 3

    # Convert array of integers to pandas series
    numbers_series = pd.Series(arr)

    # Get the window of series of
    # observations till the current time
    windows = numbers_series.expanding()

    # Create a series of moving averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    return moving_averages.tolist()

def moving_average(list, rolling_cnt):
    numbers_series = pd.Series(list)

    return numbers_series.rolling(rolling_cnt).mean().tolist()

def rising_edge(data, thresh):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos

def print_csv(heartrate_map):
    for time, heartrate in heartrate_map.items():
        print(time + ";" + str(heartrate))
