import numpy as np
import matplotlib.pyplot as plt
from numpy.f2py.capi_maps import f2cexpr


class HardDiskMC:
    def __init__(self,N,density,box_size=None,max_disp=0.1):
        """
        参数:
            N: 粒子数
            density: 数密度 (N/面积)
            box_size: 盒子边长，如果为None则根据密度计算
            max_disp: 最大位移（用于调节接受率）
        """


        self.N = N
        self.density = density
        self.max_disp = max_disp

        if box_size is None:
            self.box_size = np.sqrt(N/density)
        else:
            self.box = box_size

        self.diameter=1.0
        self.init_positions()


        self.accepted=0
        self.attempts=0

    def init_positions(self):
        """初始放在格点上"""
        n_per_side = int(np.ceil(np.sqrt(self.N)))
        spacing = self.box / n_per_side

        positions = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                if len(positions) < self.N:
                    x = (i + 0.5) * spacing
                    y = (j + 0.5) * spacing
                    positions.append([x, y])

        self.positions = np.array(positions[:self.N])

    def periodic_diff(self, r1, r2):
        """计算周期性边界下的最小镜像距离"""
        delta = r1 - r2
        # 将坐标差映射到 [-box/2, box/2] 区间
        delta = delta - self.box * np.floor(delta / self.box + 0.5)
        return delta


    def distance(self, r1, r2):
        """周期性边界下的距离"""
        diff = self.periodic_diff(r1, r2)
        return np.sqrt(np.sum(diff**2))


    def check_overlap(self,pos_new,i_particle):
        for j in range(self.N):
            if j==i_particle:
                continue
            rj=self.positions[j]
            dist=self.distance(pos_new,rj)
            if dist<self.diameter:
                return True
        return False

    def energy(self):
        """计算总能量（硬球：重叠则无穷大，否则0）"""
        for i in range(self.N):
            for j in range(i+1, self.N):
                dist = self.distance(self.positions[i], self.positions[j])
                if dist < self.diameter:
                    return np.inf
        return 0.0


    def step(self):
        self.attempts+=1

        i=np.random.randint(self.N)
        r_old=self.postions[i].copy()

        displacement=np.random.uniform(-self.max_disp, self.max_disp, 2)
        r_new=r_old + displacement

        r_new=r_new%self.box

        if self.check_overlap(r_new, i):
            return False

        self.positions[i]=r_new
        self.accepted+=1
        return True


    def run(self,n_steps,eq_steps=1000):
        print(f"开始模拟: {self.N}个粒子, 密度={self.density}, 盒子大小={self.box:.3f}")
        print(f"最大位移={self.max_disp}")

        # 先跑平衡步数
        self.accepted = 0
        self.attempts = 0

        # 生产阶段
        positions_history = []
        for step in range(n_steps):
            self.step()

            # 每10步记录一次位置（为了计算径向分布）
            if step % 10 == 0:
                positions_history.append(self.positions.copy())

        accept_rate = self.accepted / self.attempts if self.attempts > 0 else 0
        print(f"接受率: {accept_rate:.3f}")

        return accept_rate, positions_history


    def radical_distribution(self,positions_history,dr=0.02,r_max=None):
        if r_max is None:
            r_max = self.box / 2
        bins=int(r_max/dr)
        hist=np.zeros(bins)
        norm=np.zeros(bins)
        n_snapshots=len(positions_history)
        box_volume=self.box**2

        for positions in positions_history:
            for i in range(self.N):
                for j in range(i+1, self.N):
                    dist=self.distance(positions[i], positions[j])
                    if dist<r_max:
                        bin_idx=int(dist/dr)
                        if bin_idx<bins:
                            hist[bin_idx]+=2

        for i in range(bins):
            r_low=i*dr
            r_high=(i+1)*dr
            ring_area=np.pi * (r_high**2 - r_low**2)
            norm[i]+=ring_area*self.density*self.N*n_snapshots/2

        g=hist/(norm+1e-10)
        r=np.arange(bins)*dr+dr/2
        return r,g
