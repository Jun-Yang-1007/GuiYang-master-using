import time, datetime


def time_to_normal(time_unix):
    time_normal = time.gmtime(time_unix / 1000)  # 转换为普通时间格式（时间数组）
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_normal)  # 格式化为需要的格式
    return dt


def dateshift_hour(date, delta, format='%Y-%m-%d %H:%M:%S'):
    date = datetime.datetime.strptime(date, format)  # 把时间改成可按日、月、时累加形式
    target = (date + datetime.timedelta(hours=delta)).strftime(format)
    return target


def dateshift_hour2(date):
    """
    按照要求写入 2020-10-01 10:00的格式，去除秒钟
    :param date:
    :return:
    """
    target = date[0:-3]
    return target


def datetime_timestamp(dt):
    """
    将时间转化为Unix时间
    :param dt:
    :return:
    """
    new_time = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(new_time * 1000)
