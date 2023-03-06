import xml.dom.minidom

from Healthfunctions import getMinMaxHeartRate, calc_shift_len, print_csv, moving_average, moving_average_exponential, \
    rising_edge


def main():
    import_file('activity_4074255082.tcx')

def import_file(file):
    tree = xml.dom.minidom.parse(file)
    root = tree.documentElement

    trackpoints = root.getElementsByTagName('Trackpoint')
    heartrate_map = {}

    for trackpoint in trackpoints:
        time = trackpoint.getElementsByTagName('Time')[0]
        timestamp = time.firstChild.nodeValue

        heartrate_bpm = trackpoint.getElementsByTagName('HeartRateBpm')
        if heartrate_bpm.length > 0:
            heartrate = heartrate_bpm[0].getElementsByTagName('Value')[0].firstChild.nodeValue
            heartrate_map[timestamp] = int(heartrate)

    min_heartrate,max_heartrate,avg_heartrate = getMinMaxHeartRate(heartrate_map)
    #print(calc_shift_len(heartrate_map, avg_heartrate))

    moving_avg_list = moving_average(list(heartrate_map.values()), 100)
    #moving_avg_list = moving_average_exponential(list(heartrate_map.values()), .3)
    for avg in moving_avg_list:
        pass
        #print(avg)
    #print_csv(heartrate_map)

    data = rising_edge(list(heartrate_map.values()), .3)
    print(data)






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
