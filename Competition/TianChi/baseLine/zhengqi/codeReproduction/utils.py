# -*- coding:utf-8 -*-
import os
import sys

sys.path.append(os.path.pardir)


def get_file_size_mb(path: str) -> int:
    import os
    file_size = os.path.getsize(path)  # unit: B
    file_size = file_size / float(1024 * 1024)  # unit: MB
    return round(file_size, 2)  # 保留两位小数
