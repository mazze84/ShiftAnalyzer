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

    df['Heartrate_avg'] = Healthfunctions.moving_average(extract(trackpoint_map.values(), 0), 50)
    df['Heartrate_avg_rolling'] = Healthfunctions.moving_average(extract(trackpoint_map.values(), 0), 10)
    df['Speed_rolling'] = Healthfunctions.moving_average(extract(trackpoint_map.values(), 1), 5)
    print(df)
    #print(avg_dataframe)
    # Sorting the two columns in ascending order
    df.sort_values(["Time"], axis=0, inplace=True)
    #, fdf[''], fdf['']
    plt.figure(1)
    plt.subplot(211)
    plt.title('Heartrate')
    plt.plot(df.index.values, df['Heartrate'])
    plt.plot(df.index.values, df['Heartrate_avg'])
    plt.plot(df.index.values, df['Heartrate_avg_rolling'])
    plt.subplot(212)
    plt.title('Speed')
    plt.plot(df.index.values, df['Speed'])
    plt.plot(df.index.values, df['Speed_rolling'])
    #plt.plot(df.index.values, df['Distance'])
    plt.ylabel("Heartrate")
    plt.xlabel("Time")
    plt.show()

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

def extract(lst, index):
    return [item[index] for item in lst]




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
