# -*- coding: utf-8 -*-

import os

import GPUtil
import psutil
import requests

from logger import logger


class HttpUtil:
    BASE_URL = "http://172.19.20.226:8999/algorithm-server"
    UPDATE_ACTIVITY_LOG_URL = BASE_URL + "/api/algorithm/activity/log/"

    def update_activity_log(self, activity_log_id, status=None, start_time=None, end_time=None, log_obj=None,
                            output_ojb=None, cpu_use=None, gpu_use=None, memory_use=None):
        """
        更新活动日志
        :param memory_use: 内存使用量
        :param gpu_use: gpu使用量
        :param cpu_use: cpu使用量
        :param activity_log_id: 活动日志id
        :param status: 状态
        :param start_time: 开始时间
        :param end_time: 结束时间
        :param log_obj: 运行日志对象，有默认mongodb的_id
        :param output_ojb: 输出对象，有默认activity_log_id
        :return:
        """
        data = {
            "endTime": end_time,
            "logObj": log_obj,
            "outputObj": output_ojb,
            "startTime": start_time,
            "status": status,
            "cpuUse": cpu_use,
            "gpuUse": gpu_use,
            "memoryUse": memory_use
        }
        response = requests.put(self.UPDATE_ACTIVITY_LOG_URL + activity_log_id, json=data)
        if response.status_code != 200:
            logger.error("日志接口请求失败")

    def record_run_info(self, activity_log_id):
        self.update_activity_log(activity_log_id, cpu_use=self.get_cpu_info(), gpu_use=self.get_gpu_info(),
                                 memory_use=self.get_memory_info())

    def get_gpu_info(self):
        try:
            gpus = GPUtil.getGPUs()
            used = 0
            for e in gpus:
                used += e.memoryUsed
            return used * 1024 * 1024
        except:
            return None

    def get_cpu_info(self):
        pid = os.getpid()
        return int(psutil.Process(pid).cpu_percent())

    def get_memory_info(self):
        pid = os.getpid()
        return psutil.Process(pid).memory_info().rss


http_util = HttpUtil()
