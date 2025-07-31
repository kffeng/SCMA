import os
import functools
from mindspore import context
from mindspore.profiler import Profiler

_global_sync_count = 0

def get_device_id():
    device_id = os.getenv('DEVICE_ID', '0')
    return int(device_id)


def get_device_num():
    device_num = os.getenv('RANK_SIZE', '1')
    return int(device_num)