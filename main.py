import xml.dom.minidom
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
from scipy.signal import find_peaks

from datetime import datetime

import Healthfunctions

matplotlib.use('TkAgg')

def main():
    trackpoints = import_tcx_file('activity_10814740769_oacha.tcx')

    df = pd.DataFrame.from_dict(trackpoints, orient='index', columns=['Heartrate', 'Speed', 'Distance'])
    df.index.name = "Time"

    # Smooth the data using a rolling mean
    smoothed = df['Heartrate'].rolling(window=10).mean()

    # Find peaks in the smoothed data
    peaks, _ = find_peaks(smoothed, height=np.mean(smoothed), distance=10)

    #plt.plot(df.index.values, df['Heartrate'])
    #plt.plot(df.index.values, df['Heartrate'][peaks], 'x')
    #plt.show()

    df['Peaks'] = df.index.to_series()[peaks]

    df['Heartrate_avg_exponential'] = Healthfunctions.moving_average_exponential(extract(trackpoints.values(), 0), alpha=0.1, decimals=0)
    df['Heartrate_avg_rolling_10'] = Healthfunctions.moving_average(extract(trackpoints.values(), 0), 10)
    df['Speed_rolling_5'] = Healthfunctions.moving_average(extract(trackpoints.values(), 1), 5)
    df['Speed_rolling_10'] = Healthfunctions.moving_average(extract(trackpoints.values(), 1), 10)
    df['deltaT'] = df.index.to_series().diff().dt.seconds.div(60, fill_value=0)
    df.set_index('deltaT')
    #df.to_csv('activity.csv', index=True, sep=';')
    df.sort_values(["Time"], axis=0, inplace=True)
    print(df.head())
    activity_map = df.to_dict('index')

    shifts = Healthfunctions.calc_shift_len_speed(activity_map)
    rising, falling, sum = Healthfunctions.calc_rising_falling_heartrates(df)

    changes = Healthfunctions.detect_time_series_change(df)
    # print(avg_dataframe)
    # Sorting the columns in ascending order
    df_shifts = pd.DataFrame(shifts, columns=['Starttime', 'Endtime', 'AverageHeartRate', 'AverageSpeed', 'Duration', 'Active'])
    print(df_shifts.head())
    df_shifts.to_csv('shifts.csv', sep=';')
    #plot_shifts(df_shifts)
    plot_heartrate(df, changes)


    plt.show()


def plot_shifts(df_shifts):

    fig, ax = plt.subplots()

    ax.set_ylabel('Time')
    ax.set_title('Shifts')
    #ax.set_xticks(df_shifts['Duration'], df_shifts['Duration'])
    ax.bar(df_shifts['Duration'], df_shifts['AverageHeartRate'])
    plt.show()


    #plt.plot(df_shifts.index.values, df_shifts['AverageHeartRate'])
    #plt.plot(df_shifts.index.values, df_shifts['Duration'])
    #plt.plot(df_shifts.index.values, df_shifts['Starttime'])
    #plt.plot(df_shifts.index.values, df_shifts['Endtime'])

    #plt.plot(df_shifts.index.values, df_shifts['Active'])


def plot_shift_bar(df_shifts):
    fig, ax = plt.subplots()

    ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


def plot_heartrate(df, changes):
    min_hr, max_hr, avg_hr = Healthfunctions.get_min_max_avg(df['Heartrate'])
    min_speed, max_speed, avg_speed = Healthfunctions.get_min_max_avg(df['Speed'])
    spx = df['Heartrate']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Heartrate')
    plt.ylabel("Heartrate")
    plt.xlabel("Time")
    plt.axhline(y=max_hr, color='darkred', linestyle='--', label='Max heartrate')
    plt.axhline(y=avg_hr, color='orangered', linestyle='--', label='Avg heartrate')
    plt.axhline(y=min_hr, color='tomato', linestyle='--', label='Min heartrate')

    for x in changes:
        plt.axvline(df.index.values[x-1], lw=2, color='red')

    #ax.plot(df.index.values, df['Heartrate'], color='r', label='Heartrate')
    #ax.plot(df.index.values, df['Heartrate_avg_exponential'], color='r', label='Heartrate exponential')
    #ax.plot(df.index.values, df['Peaks'], color='b', label='Heartrate peaks')
    label = 0
    peaks = df['Peaks'].tolist()
    for date in peaks:
        label += 1
        ax.annotate(label, xy=(date, spx.asof(date) + 75), xytext=(date, spx.asof(date) + 255), arrowprops = dict(facecolor="black", headwidth=4, width=2, headlength=4), horizontalalignment="left", verticalalignment="top")

    #ax.fill_between(df.keys(), min_hr, df.get('Heartrate_avg_exponential'), alpha=0.7)
    ax.plot(df.index.values, df['Heartrate_avg_rolling_10'], color='firebrick', label='Heartrate rolling 10')
    ax.legend(loc=0)

    ax.set_ylim(0, max_hr)

    #plt.subplot(212)
    #plt.title('Speed')
    ax2 = ax.twinx()
    #ax2.plot(df.index.values, df['Speed'], label='Speed')
    #ax2.plot(df['Speed_rolling_10'], color='b', label='Speed rolling 10')
    ax2.plot(df['Speed_rolling_5'], color='indigo', label='Speed rolling 5')

    #plt.ylabel("Speed")
    #plt.xlabel("Time")
    ax2.legend(loc=0)
    ax2.set_ylim(0, 10)

def import_tcx_file(file):
    tree = xml.dom.minidom.parse(file)
    root = tree.documentElement

    trackpoints = root.getElementsByTagName('Trackpoint')
    trackpoint_map = {}

    heartrate = 0
    speed = 0.0
    for trackpoint in trackpoints:
        time = trackpoint.getElementsByTagName('Time')[0].firstChild.nodeValue
        timestamp = datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%f%z')
        heartrate_bpm = trackpoint.getElementsByTagName('HeartRateBpm')

        if heartrate_bpm.length > 0:
            heartrate = int(heartrate_bpm[0].getElementsByTagName('Value')[0].firstChild.nodeValue)

        distance = float(trackpoint.getElementsByTagName('DistanceMeters')[0].firstChild.nodeValue)

        tcx = trackpoint.getElementsByTagName('Extensions')[0].getElementsByTagName('ns3:TPX')
        if tcx.length > 0:
            speed_tag = tcx[0].getElementsByTagName('ns3:Speed')
            if speed_tag.length > 0:
                speed = float(speed_tag[0].firstChild.nodeValue)
        trackpoint_map[timestamp] = [heartrate, speed, distance]
    return trackpoint_map


# extracts column in matrix to list
def extract(lst, index):
    return [item[index] for item in lst]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
