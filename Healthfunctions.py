from datetime import datetime
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import ruptures as rpt

def get_min_max_avg(data):
    min_heartrate = 1000
    max_heartrate = 0
    avg = 0
    cnt = 0
    for time, trackpoint in data.items():
        heartrate = trackpoint
        avg += heartrate
        cnt += 1
        if heartrate > max_heartrate:
            max_heartrate = heartrate
        if heartrate < min_heartrate:
            min_heartrate = heartrate
    avg = calc_avg(avg, cnt)
    return min_heartrate, max_heartrate, avg


def get_peaks(df):
    smoothed = df['Heartrate_avg_rolling_10']
    peaks, _ = find_peaks(smoothed, height=np.mean(smoothed), distance=10)
    return peaks

def calc_avg(heartrate, cnt, decimals=0):
    avg = 0
    if cnt > 0:
        heartrate /= cnt
        avg = round(heartrate, decimals)
    return avg


def calc_duration_seconds(start_time, end_time):
    duration = end_time - start_time
    return duration.total_seconds()


def calc_shift_len(heartrate_map, active_heartrate=100):
    shifts = {}
    shift_cnt = 0

    avg_heartrate = 0
    heartrate_cnt = 0

    start_time = list(heartrate_map.keys())[0]
    last_heartrate = list(heartrate_map.values())[0]

    for time, heartrate in heartrate_map.items():

        is_old_heartrate_bench = last_heartrate <= active_heartrate
        is_new_heartrate_bench = heartrate <= active_heartrate
        is_old_heartrate_active = last_heartrate >= active_heartrate
        is_new_heartrate_active = heartrate >= active_heartrate
        # continue bench or active shift
        if is_old_heartrate_bench and is_new_heartrate_bench or is_old_heartrate_active and is_new_heartrate_active:
            avg_heartrate += heartrate
            heartrate_cnt += 1

        # start active shift or bench
        else:
            # save shift
            duration_last_shift = calc_duration_seconds(start_time, time)
            shifts[shift_cnt] = duration_last_shift, calc_avg(avg_heartrate, heartrate_cnt)

            # reset values
            shift_cnt += 1
            start_time = time
            avg_heartrate = heartrate
            heartrate_cnt = 1
    duration_last_shift = calc_duration_seconds(start_time, list(heartrate_map.keys())[-1])
    shifts[shift_cnt] = duration_last_shift, calc_avg(avg_heartrate, heartrate_cnt)
    return shifts


def calc_shift_len_speed(trackpoints, active_speed=.3, seconds_inactive=5):
    shifts = []
    shift_cnt = 0

    avg_heartrate = 0
    heartrate_cnt = 0
    avg_speed = 0

    start_time = list(trackpoints.keys())[0]
    last_speed = list(trackpoints.values())[0]['Speed_rolling_10']
    for time, trackpoint_list in trackpoints.items():
        heartrate = trackpoint_list['Heartrate']
        speed = trackpoint_list['Speed_rolling_10']

        is_old_speed_bench = last_speed <= active_speed
        is_new_speed_bench = speed <= active_speed
        is_old_speed_active = last_speed >= active_speed
        is_new_speed_active = speed >= active_speed

        # continue bench or active shift
        if is_old_speed_bench and is_new_speed_bench or is_old_speed_active and is_new_speed_active:
            avg_heartrate += heartrate
            heartrate_cnt += 1
            avg_speed += speed
        # state changed, wait for seconds past, but data should be saved
        # elif (is_old_speed_bench != is_new_speed_bench or is_old_speed_active != is_new_speed_active) and calc_duration_seconds(start_time, time) < seconds_inactive:
            # FIXME: save data
            # pass
        # start active shift or bench
        else:
            # save shift
            duration_last_shift = calc_duration_seconds(start_time, last_time)
            shifts.append([start_time,  last_time, calc_avg(avg_heartrate, heartrate_cnt), calc_avg(avg_speed, heartrate_cnt, decimals=2), duration_last_shift, is_old_speed_active])

            # reset values
            shift_cnt += 1
            start_time = time
            avg_heartrate = heartrate
            avg_speed = speed
            heartrate_cnt = 1
        last_time = time
        last_speed = speed
    duration_last_shift = calc_duration_seconds(start_time, list(trackpoints.keys())[-1])
    shifts.append([start_time, list(trackpoints.keys())[-1], calc_avg(avg_heartrate, heartrate_cnt), calc_avg(avg_speed, heartrate_cnt, decimals=2), duration_last_shift, (not is_old_speed_active)])
    return shifts

def detect_time_series_change(df):
    heartrates = df['Heartrate_avg_rolling_10'].to_numpy()
    # Detect the change points
    algo = rpt.Pelt(model="rbf").fit(heartrates)
    change_location = algo.predict(pen=30)
    return change_location

def calc_rising_falling_heartrates(df):
    rolling_10 = df['Heartrate_avg_rolling_10'].tolist()
    diff = np.diff(rolling_10)

    # Find the indices where the heart rate is rising and falling
    rising = np.where(diff > 0)[0]
    falling = np.where(diff < 0)[0]

    diff_sum = 0
    sum = []
    time = df.index.values[0]

    for i in range(len(diff)):
        difference = diff[i]
        if difference > 0.0 and diff_sum >= 0.0 or difference < 0.0 and diff_sum <= 0.0:
            diff_sum += difference
        elif diff_sum < 30.0 and diff_sum > -30.0:
            # change is not significant enough
            diff_sum += difference
        else:
            sum.append({time, diff_sum})
            time = df.index.values[i]
            diff_sum = difference
    print(sum)

    return rising, falling, sum

def combine_peak_with_rising_falling(df):
    # Find peaks in the heart rate data
    peaks, _ = find_peaks(df['Heartrate'])

    # Calculate the difference between consecutive heart rate values
    diff = np.diff(df['Heartrate'])

    # Find the indices where the heart rate is rising and falling
    rising = np.where(diff > 0)[0]
    falling = np.where(diff < 0)[0]

    # Filter the detected peaks to only keep those that occur during rising or falling heart rate segments
    rising_peaks = []
    falling_peaks = []
    for peak in peaks:
        if np.any((rising <= peak) & (falling > peak)):
            rising_peaks.append(peak)
        elif np.any((falling <= peak) & (rising > peak)):
            falling_peaks.append(peak)

    # Plot the heart rate data with rising and falling segments and detected peaks marked
    import matplotlib.pyplot as plt

    plt.plot(df['timestamp'], df['Heartrate'])
    plt.plot(df.index.values[rising], df['Heartrate'][rising], 'g')
    plt.plot(df.index.values[falling], df['Heartrate'][falling], 'r')
    plt.plot(df.index.values[rising_peaks], df['Heartrate'][rising_peaks], 'x', color='green')
    plt.plot(df.index.values[falling_peaks], df['Heartrate'][falling_peaks], 'x', color='red')
    plt.show()

def regression():
    # Load heart rate data into a pandas dataframe
    heart_rate_data = pd.read_csv("heart_rate_data.csv")

    # Define the independent variable (heart rate) and dependent variable (shift length)
    X = heart_rate_data["heart_rate"].values.reshape(-1, 1)
    y = heart_rate_data["shift_length"].values.reshape(-1, 1)

    # Fit a linear regression model to the data
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Use the model to predict shift length for new heart rate data
    new_heart_rate_data = pd.read_csv("new_heart_rate_data.csv")
    new_X = new_heart_rate_data["heart_rate"].values.reshape(-1, 1)
    predicted_shift_length = regressor.predict(new_X)
def arima():

    # Load heart rate data into a pandas dataframe
    heart_rate_data = pd.read_csv("heart_rate_data.csv")

    # Define the independent variable (heart rate) and dependent variable (shift length)
    X = heart_rate_data["heart_rate"].values
    y = heart_rate_data["shift_length"].values

    # Fit an ARIMA model to the data
    model = ARIMA(y, order=(1, 1, 1))
    results = model.fit()

    # Use the model to predict shift length for new heart rate data
    new_heart_rate_data = pd.read_csv("new_heart_rate_data.csv")
    new_X = new_heart_rate_data["heart_rate"].values
    predicted_shift_length = results.forecast(len(new_X))[0]

def moving_average_exponential(lst, alpha=0.3, decimals=2):
    numbers_series = pd.Series(lst)

    # Get the moving averages of series
    # of observations till the current time
    moving_averages = round(numbers_series.ewm(alpha=alpha, min_periods=1, adjust=False).mean(), decimals)

    # Convert pandas series back to list
    return moving_averages.tolist()


def moving_average_cumulative(lst, window_size):

    # Convert array of integers to pandas series
    numbers_series = pd.Series(lst)

    # Get the window of series of
    # observations till the current time
    windows = numbers_series.expanding(min_periods=1)

    # Create a series of moving averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    return moving_averages.tolist()


def moving_average(lst, rolling_cnt):
    numbers_series = pd.Series(lst)
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
