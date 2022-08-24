import numpy

class Calculation:
    def __init__(self):
        pass

    def calcu_regional(self, data_var, index_time_1, index_time_2, index_lat_1, index_lat_2, index_lon_1, index_lon_2): #Calculate the sum and mean value in a region
        sum = numpy.nansum(data_var[index_time_1 : index_time_2+1, index_lat_1 : index_lat_2 + 1, index_lon_1 : index_lon_2 + 1])
        mean = numpy.nanmean(data_var[index_time_1 : index_time_2+1, index_lat_1 : index_lat_2 + 1, index_lon_1 : index_lon_2 + 1])
        return sum, mean

    def calcu_bord_index(self, data_lon, data_lat, bord_n, bord_s, bord_w, bord_e):     #calculate the index of the lat and lon based on the border of the region
        array_lat_s_1 = numpy.argwhere(data_lat <= bord_s)
        array_lat_s_2 = numpy.argwhere(data_lat > bord_s)
        if (bord_s - array_lat_s_1[-1]) < (array_lat_s_2[0] - bord_s):  #normally, the border of region will be between two latitudes, then calculate the distance to make sure which one should we choose
            index_s = array_lat_s_1[-1]
        else:
            index_s = array_lat_s_2[0]

        array_lat_n_1 = numpy.argwhere(data_lat <= bord_n)
        array_lat_n_2 = numpy.argwhere(data_lat > bord_n)
        if (bord_n - array_lat_n_1[-1]) <= (array_lat_n_2[0] - bord_n):
            index_n = array_lat_n_1[-1]
        else:
            index_n = array_lat_n_2[0]

        array_lon_w_1 = numpy.argwhere(data_lon <= bord_w)
        array_lon_w_2 = numpy.argwhere(data_lon > bord_w)
        if (bord_w - array_lon_w_1[-1]) < (array_lon_w_2[0] - bord_w):
            index_w = array_lon_w_1[-1]
        else:
            index_w = array_lon_w_2[0]

        array_lon_e_1 = numpy.argwhere(data_lon <= bord_e)
        array_lon_e_2 = numpy.argwhere(data_lon > bord_e)
        if (bord_e - array_lon_e_1[-1]) <= (array_lon_e_2[0] - bord_e):
            index_e = array_lon_e_1[-1]
        else:
            index_e = array_lon_e_2[0]

        index_s = index_s[0]    #convert from array to integer
        index_n = index_n[0]
        index_w = index_w[0]
        index_e = index_e[0]

        return index_s, index_n, index_w, index_e