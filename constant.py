import datetime
from time_convert import dateshift_hour
from time_convert import datetime_timestamp

finish_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
start_time = dateshift_hour(finish_time, -7000)


class Constant:
    headers = {'ignore-token': 'true', 'Content-Type': 'application/json'}
    get_base = 'http://wgms.dev.fpi-inc.site/wgms-forecast-server/api/v1.0/client/measure_data/query?mn='
    post_base = 'http://wgms.dev.fpi-inc.site/wgms-forecast-server/api/v1.0/client/forecast_data/upload'
    siteid = ['2000', '1011', '1009']
    begin_time = '&beginTime='
    begin_time1 = str(datetime_timestamp(start_time))
    end_time = '&endTime='
    end_time1 = str(datetime_timestamp(finish_time))
    factor = ['w21011', 'w21003', 'w21001']
    finish_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    start_time = dateshift_hour(finish_time, -7000)


constants = Constant()
