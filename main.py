#!/usr/bin/env python
# -*- coding:utf-8 -*-
import time, datetime
import numpy
import requests
from model import base_model
from sklearn.preprocessing import StandardScaler
import warnings
import json
import pandas as pd
import random
from apscheduler.schedulers.blocking import BlockingScheduler
import matplotlib.pyplot as plt
from http_util import http_util
random.seed(7)
warnings.filterwarnings('ignore')
import tensorflow as tf
from constant import constants
from time_convert import dateshift_hour, dateshift_hour2
from save_db import insert_output
from achieve_params import Flags,command_params
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)

predict_length = Flags.predict_length
look_back = Flags.look_back
look_after = 1

scaler = StandardScaler()


def reindex_dataframe(df, start_time=None, end_time=None, freq='1H'):
    if start_time is None:
        start_time = df.index.min()
    if end_time is None:
        end_time = df.index.max()
    print(start_time, end_time)
    df = df.reindex(pd.date_range(start=start_time, end=end_time, freq=freq))
    return df


def process_data(dataset):
   # data = dataset.fillna(method='ffill')
    data = dataset.ffill().bfill()
    dataset = data.astype('float32')
    dataset = numpy.array(dataset)
    dataset = dataset.reshape(-1, 1)
    dataset = scaler.fit_transform(dataset)
    return dataset


def create_dateset(dataset, look_back=1, look_after=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        x = dataset[i:(i + look_back), 0]
        y = dataset[(i + look_back):(i + look_after + look_back), 0]
        dataX.append(x)
        dataY.append(y)
    return numpy.array(dataX), numpy.array(dataY)


def get_train_set(dataset, scale=1):
    train_size = int(len(dataset) * scale)
    train = dataset[0:train_size, :]
    trainX, trainY = create_dateset(train, look_back, look_after)
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    print(trainX.shape, trainY.shape)
    return trainX, trainY


def get_forecast_input(trainx, trainy):
    trainx = trainx[trainx.shape[0] - 1][trainx.shape[1] - 1]
    trainy = trainy[trainy.shape[0] - 1]
    input = []

    for i in range(look_after, look_back):
        input.append(trainx[i])
    for i in range(look_after):
        input.append(trainy[i])
    input = numpy.reshape(input, (1, 1, look_back))
    return input


def get_next_day(model, input, forecast):
    second_day_input = get_forecast_input(input, forecast)
    second_day_forecast = model.predict(second_day_input)
    second_day_forecast = scaler.inverse_transform(second_day_forecast)
    return {"input": second_day_input, "forecast": second_day_forecast}


def forecast(data):
    data = process_data(data)
    trainx, trainy = get_train_set(data)
    model, history = base_model(trainx, trainy, input_dim=look_back, output_dim=look_after, type='easy')  # epoch=200
    # train_predict = scaler.inverse_transform(model.predict(trainx))
    # trainy = scaler.inverse_transform(trainy)

    # plt.figure(dpi=300)
    # plt.plot(train_predict[100:300], label='train_predict_data', linewidth='0.8', c='red')  # 将不同结果放在一起，一个真实值，两种预测值
    # plt.plot(trainy[100:300], label='train_raw_data', linewidth='0.8', c='green')
    # plt.legend()
    # plt.show()

    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend(['train', 'test'])
    # plt.legend()
    # plt.show()

    forecast_result = []
    first_day_input = get_forecast_input(trainx, trainy)
    first_day_forecast = model.predict(first_day_input)
    array = []
    array.append({"input": first_day_input, "forecast": first_day_forecast})
    for i in range(predict_length):
        array.append(get_next_day(model, array[-1]['input'], array[-1]['forecast']))

        forecast_result.append(array[i]['forecast'][0][0])
    print(forecast_result)
    return forecast_result


def get_data(url):
    response = requests.get(url, headers=constants.headers)
    response.encoding = 'utf-8'
    text = response.text
    # print('text', text)
    new_text = json.loads(text)['rows']
    raw_df = pd.DataFrame.from_dict(new_text, orient='columns')
    raw_df = raw_df.reset_index(drop=True)
    return raw_df


def post_data(url, body):
    request = requests.session()
    response = request.post(url, data=json.dumps(body), headers=constants.headers)
    print(response.status_code)
    print(response.text)


def gen_body(site, time, many_factor_data):
    body = {
        'mn': site,
        'riverOrLake': '1',
        'dataSource': '006',
        'dataTimeType': 'hour',
        'monitorTime': time,
        'pollutants': []
    }
    pollutants = []
    for i, val in enumerate(many_factor_data):
        pollutants.append({"code": constants.factor[i], "value": str(val)})
    body["pollutants"] = pollutants
    return body


collection = []


def main():
    for site in constants.siteid:
        site_result = []
        for factor in constants.factor:
            print(site, factor)
            get_url = constants.get_base + site + constants.begin_time + constants.begin_time1 + constants.end_time + \
                      constants.end_time1 + \
                      '&dataTimeType=hour' + '&factorCodes=' + factor
            raw_df = get_data(get_url)
            print(raw_df)
            factor_result = forecast(raw_df[factor])  # 按因子获取预测值
            #  site_result.append({"factor": factor, "data": factor_result})
            site_result.append(factor_result)
        site_result2 = numpy.array(site_result).T
        print(numpy.array(site_result).T)
        print(site_result)
        print("site --------------->", site)

        for i in range(predict_length):
            normal_time = constants.finish_time
            time = dateshift_hour(normal_time, i)  # 单个因子预测完7小时候，换另外因子
            time = dateshift_hour2(time)
            print('time is ', time)
            body = gen_body(str(site), str(time), site_result2[i])
            '''
            进入数据中台的输出和日志中
            '''
            # output = {'activity_log_id': Flags.activity_log_id, 'output': body}
            # insert_output(output)
            # http_util.record_run_info(command_params.activity_log_id)
            collection.append(body)
            print('body', body)
            post_url = constants.post_base
        post_data(post_url, collection)


if __name__ == '__main__':
    # scheduler = BlockingScheduler()
    # scheduler.add_job(main, 'cron', hour='0-23', minute='00',  misfire_grace_time=2400)
    # scheduler.start()
    main()
