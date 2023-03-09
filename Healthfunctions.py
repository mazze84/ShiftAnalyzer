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
    avg = calc_avg(avg, cnt)
    return min_heartrate,max_heartrate,avg

def calc_avg(heartrate, cnt, decimals=0):
    avg = 0
    if cnt > 0:
        heartrate /= cnt
        avg = round(heartrate, decimals)
    return avg

def calc_duration_seconds(start_time, end_time):
    duration = end_time - start_time
    return duration.total_seconds()

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
            duration_last_shift = calc_duration_seconds(start_time, time)
            shifts[shift_cnt] = duration_last_shift, calc_avg(avg_heartrate, heartrate_cnt)

            # reset values
            shift_cnt += 1
            start_time = time
            avg_heartrate = heartrate
            heartrate_cnt = 1
    duration_last_shift = calc_duration_seconds(start_time, list(heartrateMap.keys())[-1])
    shifts[shift_cnt] = duration_last_shift, calc_avg(avg_heartrate, heartrate_cnt)
    return shifts

def calc_shift_len_speed(trackpoint_map, active_speed=.3, seconds_inactive=5):
    shifts = {}
    shift_cnt = 0

    avg_heartrate = 0
    heartrate_cnt = 0
    avg_speed = 0
    print(trackpoint_map)

    start_time = list(trackpoint_map.keys())[0]
    last_speed = list(trackpoint_map.values())[0]['Speed_rolling']
    is_new_speed_active = False
    for time, trackpoint_list in trackpoint_map.items():
        heartrate = trackpoint_list['Heartrate']
        speed = trackpoint_list['Speed_rolling']

        is_old_speed_bench = last_speed <= active_speed
        is_new_speed_bench = speed <= active_speed
        is_old_speed_active = last_speed >= active_speed
        is_new_speed_active = speed >= active_speed

        # continue bench or active shift
        if (is_old_speed_bench and is_new_speed_bench or is_old_speed_active and is_new_speed_active) or (is_old_speed_bench != is_new_speed_bench or is_old_speed_active != is_new_speed_active) and calc_duration_seconds(start_time, time) < seconds_inactive:
            avg_heartrate += heartrate
            heartrate_cnt += 1
            avg_speed += speed
        # start active shift or bench
        else:
            # save shift
            duration_last_shift = calc_duration_seconds(start_time, last_time)
            shifts[shift_cnt] = str(start_time), str(last_time), calc_avg(avg_heartrate, heartrate_cnt), calc_avg(avg_speed, heartrate_cnt, decimals=2), duration_last_shift, is_new_speed_active

            # reset values
            shift_cnt += 1
            start_time = time
            avg_heartrate = heartrate
            avg_speed = speed
            heartrate_cnt = 1
        last_time = time
        last_speed = speed
    duration_last_shift = calc_duration_seconds(start_time, list(trackpoint_map.keys())[-1])
    shifts[shift_cnt] = start_time, list(trackpoint_map.keys())[-1], calc_avg(avg_heartrate, heartrate_cnt), calc_avg(avg_speed, heartrate_cnt, decimals=2), duration_last_shift, not is_new_speed_active
    return shifts

def moving_average_exponential(list, alpha=0.3, decimals=2):
    numbers_series = pd.Series(list)

    # Get the moving averages of series
    # of observations till the current time
    moving_averages = round(numbers_series.ewm(alpha=alpha, min_periods=1, adjust=False).mean(), decimals)

    # Convert pandas series back to list
    return moving_averages.tolist()

def moving_average_cumulative(list, window_size):

    # Convert array of integers to pandas series
    numbers_series = pd.Series(list)

    # Get the window of series of
    # observations till the current time
    windows = numbers_series.expanding(min_periods=1)

    # Create a series of moving averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    return moving_averages.tolist()

def moving_average(list, rolling_cnt):
    numbers_series = pd.Series(list)

    return numbers_series.rolling(rolling_cnt, min_periods=1).mean().tolist()

def rising_edge(data, thresh):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos

def print_csv(heartrate_map):
    for time, heartrate in heartrate_map.items():
        print(time + ";" + str(heartrate))

# extracts column in matrix to list
def extract(lst, index):
    return [item[index] for item in lst]