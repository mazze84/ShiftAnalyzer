from datetime import datetime
import numpy as np

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


def moving_average_exponential(arr, x):
    i = 1
    # Initialize an empty list to
    # store exponential moving averages
    moving_averages = []

    # Insert first exponential average in the list
    moving_averages.append(arr[0])

    # Loop through the array elements
    while i < len(arr):
        # Calculate the exponential
        # average by using the formula
        window_average = round((x * arr[i]) + (1 - x) * moving_averages[-1], 2)

        # Store the cumulative average
        # of current window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages

def moving_average(heartrate_list, rolling_cnt):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(heartrate_list) - rolling_cnt + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = heartrate_list[i: i + rolling_cnt]

        # Calculate the average of current window
        window_average = round(sum(window) / rolling_cnt, 0)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return moving_averages

def rising_edge(data, thresh):
    sign = data >= thresh
    pos = np.where(np.convolve(sign, [1, -1]) == 1)
    return pos

def print_csv(heartrate_map):
    for time, heartrate in heartrate_map.items():
        print(time + ";" + str(heartrate))
