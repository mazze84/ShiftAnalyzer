import xml.dom.minidom
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


import pandas as pd


def main():
    heartrate_map = import_tcx_file('activity_10630586552.tcx')
    # loading 10 rows of the file
    fdf = pd.DataFrame.from_dict(heartrate_map, orient='index', columns=['Heartrate', 'Speed'])

    # Sorting the two columns in ascending order
    fdf.sort_values(["Heartrate"],
                    axis=0,
                    inplace=True)

    heartrate_stats = fdf['Heartrate']
    speed_stats = fdf['Speed']

    plt.plot(heartrate_stats, speed_stats)
    plt.show()

def import_tcx_file(file):
    tree = xml.dom.minidom.parse(file)
    root = tree.documentElement

    trackpoints = root.getElementsByTagName('Trackpoint')
    heartrate_map = {}

    for trackpoint in trackpoints:
        time = trackpoint.getElementsByTagName('Time')[0]
        timestamp = time.firstChild.nodeValue

        heartrate_bpm = trackpoint.getElementsByTagName('HeartRateBpm')

        heartrate = 0
        speed = 0
        if heartrate_bpm.length > 0:
            heartrate = int(heartrate_bpm[0].getElementsByTagName('Value')[0].firstChild.nodeValue)

        tcx = trackpoint.getElementsByTagName('Extensions')[0].getElementsByTagName('ns3:TPX')
        if tcx.length > 0:
            speed_tag = tcx[0].getElementsByTagName('ns3:Speed')
            if speed_tag.length > 0:
                speed = float(speed_tag[0].firstChild.nodeValue)
        if (heartrate > 0 or speed > 0):
            heartrate_map[timestamp] = {heartrate, speed}
    return heartrate_map






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
