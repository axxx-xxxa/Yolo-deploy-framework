import sys
import time

import numpy as np
import psutil
from pynvml import *
import matplotlib.pyplot as plt
import time

# 设置中文语言
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

def nvidia_info():
    # pip install nvidia-ml-py
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        nvmlInit()
        nvidia_dict["nvidia_version"] = nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            nvmlShutdown()
        except:
            pass
    return nvidia_dict

def Memlog(p):
    # print('当前进程的内存使用：%.2f GB' % (p.memory_info().rss / 1024 / 1024 / 1024))
    # print('当前进程的内存使用：%.2f MB' % (p.memory_info().rss / 1024 / 1024))
    # print('显存总计：        %.2f GB' % (nvidia_info()['gpus'][0]['total'] / 1024 / 1024 / 1024))
    # print('显存使用：        %.2f MB' % (nvidia_info()['gpus'][0]['used'] / 1024 / 1024))
    # print('显存使用：        %.2f GB' % (nvidia_info()['gpus'][0]['used'] / 1024 / 1024 / 1024))
    # print('显存剩余：        %.2f GB' % (nvidia_info()['gpus'][0]['free'] / 1024 / 1024 / 1024))
    # print('\n')
    # time.sleep(interval)
    return "%.2f" % (p.memory_info().rss / 1024 / 1024 / 1024),\
            "%.2f" % (nvidia_info()['gpus'][0]['used'] / 1024 / 1024 / 1024)


class Memfig:
    def __init__(self, interval, figX):
        self.fig = plt.figure(dpi=80, figsize=(16, 8))
        self.interval = interval
        self.titles = ["CpuMem", "GpuMem"]
        self.line = None
        self.obsX = []
        self.obsY1 = []
        self.obsY2 = []
        self.figX = figX
        self.grid = [self.fig.add_subplot(2, 1, 1),
                    self.fig.add_subplot(2, 1, 2), ]
        # 给表的Y轴位置加上标签，rotation代表让文字横着展示，labelpad代表文字距表格多远了
        for t, g in zip(self.titles, self.grid):
            g.set_ylabel(t, rotation=0, fontsize=12, labelpad=40)



    def update(self, CM, GM, duration):
        self.obsX.append(duration + self.figX)
        self.obsY1.append(float(CM))
        self.obsY2.append(float(GM))
        if self.line is None:
            line1 = self.grid[0].plot(self.obsX, self.obsY1, '-r')[0]
            line2 = self.grid[1].plot(self.obsX, self.obsY2, '-b')[0]

        line1.set_xdata(self.obsX)
        line2.set_xdata(self.obsX)
        line1.set_ydata(self.obsY1)
        line2.set_ydata(self.obsY2)

        self.grid[0].set_ylim([min(self.obsY1) // 2, max(self.obsY1) + 2])
        self.grid[1].set_ylim([min(self.obsY2) // 2, max(self.obsY2) + 2])

        plt.pause(self.interval)



    def record(self):
        Cmax = str(max(self.obsY1) * 1024)
        Cmin = str(min(self.obsY1) * 1024)
        Gmax = str(max(self.obsY2) * 1024)
        Gmin = str(min(self.obsY2) * 1024)
        Cavg = str(np.average(self.obsY1) * 1024)
        Gavg = str(np.average(self.obsY2) * 1024)
        Cstr = f"CPU Memory MIN : {Cmin}MB MAX : {Cmax}MB AVG : {Cavg}MB"
        Gstr = f"GPU Memory MIN : {Gmin}MB MAX : {Gmax}MB AVG : {Gavg}MB"

        self.grid[0].set_title(Cstr)
        self.grid[1].set_title(Gstr)

def refresh(fig, figname):
    fig.record()
    plt.savefig(figname)
    plt.close()

interval, epoch, figX = 0.1, 1, 0
step_time = 1 * 3 * 60 # 3分钟一张图
hold_time = 10 * 60 * 60 # 1小时后退出
# hold_time = 10 * 60 * 60 # 10小时后退出

if __name__ == '__main__':


    if len(sys.argv) < 2:
        print ("missing pid arg")
    # sys.exit()

    pid = int(sys.argv[1])

    try:
        p = psutil.Process(pid)
    except:
        print ("no pid found")



    fig = Memfig(interval, figX)
    st, ct = time.time(), time.time()
    while True:
        try:
            if time.time() - st > hold_time:
                refresh(fig, f"figs/Mem_Epoch_last.png")
                break

            if time.time() - ct < step_time:
                fig.update(*Memlog(p), time.time() - ct)

            else:
                refresh(fig, f"figs/Mem_Epoch_{epoch}.png")
                figX += (time.time() - ct)
                fig = Memfig(interval, figX)
                ct = time.time()
                epoch += 1
        except:
            refresh(fig, f"figs/Mem_Epoch_break.png")
            break

