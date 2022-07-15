%addpath(genpath('D:\Tower_result\Global_simulations'))
Map = shaperead('IPCC-WGI-reference-regions-v4.shp')
surface_data = 'surfdata_0.9x1.25_hist_78pfts_CMIP6_simyr2000_c201126.nc'
LONGXY = ncread(surface_data,'LONGXY')
LATIXY = ncread(surface_data,'LATIXY')

str_PCT_CFT = 'PCT_CFT.nc'
data_PCT_CFT   = ncread(str_PCT_CFT,'Runoff')

percent = 10
data_PCT_CFT(data_PCT_CFT<percent) = 0 
data_PCT_CFT(data_PCT_CFT>0) = 1

[row,col] = find(data_PCT_CFT)       %get the row and col of the grid with more than 20 percentage
x = LONGXY(row, col)
x = x(:, 1)
x_unchanged = x
x(x>180) = x(x>180)-360
y = LATIXY(row, col)
y = y(1, :)

str_variables = ["EFLX_LH_TOT","FSH","LWup","SWup","Qle","QVEGT","QRUNOFF","TOTSOILLIQ"]
str_noi = 'NOIRR\'
str_ctl = 'CTL\'
str_irr = 'IRR_satu\'
str_mon = ["_mon01", "_mon02", "_mon03", "_mon04", "_mon05", "_mon06", "_mon07", "_mon08","_mon09", "_mon10", "_mon11", "_mon12"]

sum_noi = zeros(59,12,8)
sum_ctl = zeros(59,12,8)
sum_irr = zeros(59,12,8)

for v = 1 : 8
    for m = 1 : 12
        str_noi_variablas = strcat(str_noi, str_variables(v), str_mon(m), '.nc_timemean')
        str_ctl_variablas = strcat(str_ctl, str_variables(v), str_mon(m), '.nc_timemean')
        str_irr_variablas = strcat(str_irr, str_variables(v), str_mon(m), '.nc_timemean')

        data_noi_variablas = ncread(str_noi_variablas, str_variables(v))
        data_ctl_variablas = ncread(str_ctl_variablas, str_variables(v))
        data_irr_variablas = ncread(str_irr_variablas, str_variables(v))

        for i = 1 : 2451
            if ~isnan(data_noi_variablas(row(i), col(i))) && ~isnan(data_irr_variablas(row(i), col(i))) && ~isnan(data_ctl_variablas(row(i), col(i)))
                sum_noi(1,m,v) = sum_noi(1,m,v) + data_noi_variablas(row(i), col(i))
                sum_ctl(1,m,v) = sum_ctl(1,m,v) + data_ctl_variablas(row(i), col(i))
                sum_irr(1,m,v) = sum_irr(1,m,v) + data_irr_variablas(row(i), col(i))
            end
        end
        sum_noi(1,m,v) = sum_noi(1,m,v) / 2451
        sum_ctl(1,m,v) = sum_ctl(1,m,v) / 2451
        sum_irr(1,m,v) = sum_irr(1,m,v) / 2451
        for i = 1 : 58
            if i == 17 || i == 18 || i == 20 || i == 21 || i ==22
                x_bord = Map(i).X
                %x_bord(x_bord<0) = x_bord(x_bord<0)+360
                y_bord = Map(i).Y
                [in,on] = inpolygon(x,y,x_bord,y_bord)
                row1 = row(in)
                col1 = col(in)

                row2 = row(on)
                col2 = col(on)

                row_fi = [row1;row2]
                col_fi = [col1;col2]

                line = size(row_fi)
                line = line(1)
                num = 0
                for j = 1 : line        
                    if ~isnan(data_noi_variablas(row_fi(j), col_fi(j))) && ~isnan(data_irr_variablas(row_fi(j), col_fi(j))) && ~isnan(data_ctl_variablas(row_fi(j), col_fi(j)))              
                        sum_noi(i+1,m,v) = data_noi_variablas(row_fi(j), col_fi(j)) + sum_noi(i+1,m,v)
                        sum_ctl(i+1,m,v) = data_ctl_variablas(row_fi(j), col_fi(j)) + sum_ctl(i+1,m,v)
                        sum_irr(i+1,m,v) = data_irr_variablas(row_fi(j), col_fi(j)) + sum_irr(i+1,m,v)
                        num = num + 1
                    end
                end
                sum_noi(i+1,m,v) = sum_noi(i+1,m,v) / num
                sum_ctl(i+1,m,v) = sum_ctl(i+1,m,v) / num
                sum_irr(i+1,m,v) = sum_irr(i+1,m,v) / num
            else
                x_bord = Map(i).X
                %x_bord(x_bord<0) = x_bord(x_bord<0)+360
                y_bord = Map(i).Y
                [in,on] = inpolygon(x,y,x_bord,y_bord)
                row1 = row(in)
                col1 = col(in)

                row2 = row(on)
                col2 = col(on)

                row_fi = [row1;row2]
                col_fi = [col1;col2]

                line = size(row_fi)
                line = line(1)
                num = 0
                for j = 1 : line        
                    if ~isnan(data_noi_variablas(row_fi(j), col_fi(j))) && ~isnan(data_irr_variablas(row_fi(j), col_fi(j))) && ~isnan(data_ctl_variablas(row_fi(j), col_fi(j)))              
                        sum_noi(i+1,m,v) = data_noi_variablas(row_fi(j), col_fi(j)) + sum_noi(i+1,m,v)
                        sum_ctl(i+1,m,v) = data_ctl_variablas(row_fi(j), col_fi(j)) + sum_ctl(i+1,m,v)
                        sum_irr(i+1,m,v) = data_irr_variablas(row_fi(j), col_fi(j)) + sum_irr(i+1,m,v)
                        num = num + 1
                    end
                end
                sum_noi(i+1,m,v) = sum_noi(i+1,m,v) / num
                sum_ctl(i+1,m,v) = sum_ctl(i+1,m,v) / num
                sum_irr(i+1,m,v) = sum_irr(i+1,m,v) / num
            end
        end
    end
end





