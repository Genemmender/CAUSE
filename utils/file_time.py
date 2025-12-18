import os
import time


class FileTime:
    def __init__(self, path):
        self.path = path
        self.create_time = os.path.getctime(path)  # type: float
        self.current_time = time.time()  # type: float
        self.interval = self.current_time - self.create_time

    @staticmethod
    def _get_readable_time(t: float) -> str:
        """
        将时间戳转换为人类可读的格式。
        :param t: 时间戳
        :return: 格式化后的时间字符串
        """
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))

    @property
    def readable_create_time(self) -> str:
        return self._get_readable_time(self.create_time)

    @property
    def readable_interval(self) -> str:
        # 一周前，X天前，X小时前，X分钟前，刚刚
        if self.interval > 7 * 24 * 60 * 60:
            return f'{self.interval / (7 * 24 * 60 * 60):.0f}周前'
        elif self.interval > 24 * 60 * 60:
            return f'{self.interval / (24 * 60 * 60):.0f}天前'
        elif self.interval > 60 * 60:
            return f'{self.interval / (60 * 60):.0f}小时前'
        elif self.interval > 60:
            return f'{self.interval / 60:.0f}分钟前'
        else:
            return '刚刚'
