import os
import torch
from pigmento import pnt  # pigmento 是一个用于打印信息的库，pnt 可理解为 print 的美化版本


class GPU:
    @classmethod
    def parse_gpu_info(cls, line, args):
        """
        将 nvidia-smi 的一行输出解析成字典格式，字段名为 args 中的 key。
        自动将 memory/power 数值单位转换为 float 或 int。
        """

        def to_number(v):
            # 提取数值，去掉单位如 MiB / W，并转为 float
            return float(v.upper().strip().replace('MIB', '').replace('W', ''))

        def processor(k, v):
            # 如果值为 'Not Support'，保留原样，否则转换为数字（int）
            return (int(to_number(v)) if 'Not Support' not in v else 1) if k in params else v.strip()

        params = ['memory.free', 'memory.total', 'power.draw', 'power.limit']
        return {k: processor(k, v) for k, v in zip(args, line.strip().split(','))}

    @classmethod
    def get_gpus(cls):
        """
        使用 nvidia-smi 命令获取所有 GPU 的状态信息。
        返回一个包含每个 GPU 字典的列表。
        """
        args = ['index', 'gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(args))
        results = os.popen(cmd).readlines()
        return [cls.parse_gpu_info(line, args) for line in results]

    @classmethod
    def get_maximal_free_gpu(cls):
        """
        返回拥有最大空闲显存（memory.free）的 GPU 的空闲显存大小（单位：MiB）。
        """
        gpus = cls.get_gpus()
        gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
        return int(gpu['memory.free'])

    @classmethod
    def auto_choose(cls, torch_format=False):
        """
        自动选择最适合使用的设备：
        1. 优先选择空闲显存最大的 CUDA GPU；
        2. 若无 CUDA，尝试使用 macOS 的 MPS；
        3. 否则回退至 CPU。

        参数：
            torch_format: 如果为 True，则返回 "cuda:0" / "mps" / "cpu" 这样的字符串；
                          否则返回 GPU index 或 -1（非 GPU）。

        返回：
            str 或 int：根据 torch_format 返回设备标识。
        """
        if torch.cuda.is_available():
            gpus = cls.get_gpus()
            chosen_gpu = sorted(gpus, key=lambda d: d['memory.free'], reverse=True)[0]
            pnt('choose', chosen_gpu['index'], 'GPU with',
                chosen_gpu['memory.free'], '/', chosen_gpu['memory.total'], 'MB')
            if torch_format:
                return "cuda:" + str(chosen_gpu['index'])
            return int(chosen_gpu['index'])
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pnt('MPS available: using mac M series GPU')
            if torch_format:
                return "mps"
            return -1
        else:
            pnt('not support cuda or mps, switch to CPU')
            if torch_format:
                return "cpu"
            return -1


if __name__ == '__main__':
    # 示例代码：自动选择设备，并打印结果
    device = GPU.auto_choose(torch_format=True)
    pnt("Selected device:", device)