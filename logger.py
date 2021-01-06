# -*- coding: utf-8 -*-
import logging
from logging import handlers

logger = logging.getLogger("log/api.log")
fmt = logging.Formatter(fmt="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
# 文件
file_handler = handlers.TimedRotatingFileHandler(filename="log/api.log", when="D", encoding="utf-8")
file_handler.setFormatter(fmt)
# 控制台
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
