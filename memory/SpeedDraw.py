import os
import matplotlib.pyplot as plt
import sys
import numpy as np
# 设置中文语言
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False


class Speedfig:
    def __init__(self, txt_path):
        self.name = txt_path.split(".")[0]
        with open(txt_path, "r") as f:
            tlines = f.readlines()
        self.tlines = tlines

    def init(self):
        self.fig = plt.figure(dpi=80, figsize=(16, 12))
        self.grids = [self.fig.add_subplot(4, 1, 1),
                      self.fig.add_subplot(4, 1, 2),
                      self.fig.add_subplot(4, 1, 3),
                      self.fig.add_subplot(4, 1, 4)]
        self.titles = ["Init", "FirstInfer", "LoopInfer", "Release"]
        for t, g in zip(self.titles, self.grids):
            g.set_ylabel(t, rotation=0, fontsize=12, labelpad=40)

    def make(self, hasFirst):
        self.init()

        lines = [None, None, None, None]
        colors = ['-r', '-g', '-b', '-b']
        self.obsX = []
        self.obsY = [[], [], [], []]

        self.tlines = self.tlines if hasFirst else self.tlines[1:]
        plt_name = f"{self.name}_First.png" if hasFirst else f"{self.name}.png"
        for epoch, tline in enumerate(self.tlines):

            self.obsX.append(epoch)
            for s, y, line in zip(tline.strip().split(" "), self.obsY, lines):
                y.append(float(s))
            for color, line, grid, y in zip(colors, lines, self.grids, self.obsY):
                line = grid.plot(self.obsX, y, color)[0]
                line.set_xdata(self.obsX)
                line.set_ydata(y)

        self.record()
        plt.savefig(plt_name)
        plt.close()

    def record(self):
        for grid, y in zip(self.grids,self.obsY):
            Sstr = "SPEED MIN : %.2f ms MAX : %.2f ms AVG : %.2f ms" % (min(y),max(y),np.average(y))
            grid.set_title(Sstr)

    def forward(self):
        self.make(1)
        self.make(0)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("missing txt arg")

    # txt="chengkou.txt"
    Speedfig(sys.argv[1]).forward()


    # exit()
