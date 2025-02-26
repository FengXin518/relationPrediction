import os
import time


def Current_time():
    # 获取当前时间戳并增加8小时
    adjusted_timestamp = time.time() + 8 * 3600
    # 转换时间戳并格式化
    formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(adjusted_timestamp))
    return formatted_time 