import xml.dom.minidom
import matplotlib

from datetime import datetime

import Healthfunctions

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    trackpoint_map = import_tcx_file('activity_10630586552.tcx')

    # loading 10 rows of the file
    df = pd.DataFrame.from_dict(trackpoint_map, orient='index', columns=['Heartrate', 'Speed', 'Distance'])
    df.index.name = "Time"

    df['Heartrate_avg_exponential'] = Healthfunctions.moving_average_exponential(extract(trackpoint_map.values(), 0), alpha=0.1, decimals=0)
    df['Heartrate_avg_rolling_10'] = Healthfunctions.moving_average(extract(trackpoint_map.values(), 0), 10)
    df['Speed_rolling_10'] = Healthfunctions.moving_average(extract(trackpoint_map.values(), 1), 10)
    print(df)
    df.to_csv('activity.csv', index=True, sep=';')

    activity_map = df.to_dict('index')

    shifts = Healthfunctions.calc_shift_len_speed(activity_map)

    # print(avg_dataframe)
    # Sorting the columns in ascending order
    df.sort_values(["Time"], axis=0, inplace=True)
    plot_heartrate(df)
    df_shifts = pd.DataFrame.from_dict(shifts, orient='index', columns=['Starttime', 'Endtime', 'AverageHeartRate', 'AverageSpeed', 'Duration', 'Active'])
    print(df_shifts)

    #plot_shifts(shifts)
    plt.show()


def plot_shifts(df_shifts):
    #plt.figure(2)

    plt.plot(df_shifts.index.values, df_shifts['Starttime'])
    plt.plot(df_shifts.index.values, df_shifts['Endtime'])
    plt.plot(df_shifts.index.values, df_shifts['AverageHeartRate'])
    plt.plot(df_shifts.index.values, df_shifts['AverageSpeed'])
    plt.plot(df_shifts.index.values, df_shifts['Duration'])
    plt.ylabel("Heartrate")
    plt.xlabel("Shift")


def plot_shift_bar(df_shifts):
    fig, ax = plt.subplots()

    ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)

    ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
           ylim=(0, 8), yticks=np.arange(1, 8))

    plt.show()


def plot_heartrate(df):
    plt.figure(1)
    plt.subplot(211)
    plt.title('Heartrate')
    plt.ylabel("Heartrate")
    plt.xlabel("Time")
    plt.plot(df.index.values, df['Heartrate'], label='Heartrate')
    plt.plot(df.index.values, df['Heartrate_avg_exponential'], label='Heartrate exponential')
    plt.plot(df.index.values, df['Heartrate_avg_rolling_10'], label='Heartrate rolling 10')
    plt.legend()
    plt.subplot(212)
    plt.title('Speed')
    plt.plot(df.index.values, df['Speed'], label='Speed')
    plt.plot(df.index.values, df['Speed_rolling_10'], label='Speed rolling 10')
    plt.ylabel("Speed")
    plt.xlabel("Time")
    plt.legend()


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
