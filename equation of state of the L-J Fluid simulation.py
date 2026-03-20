import random
import math
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（避免可视化时中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

### 核心参数设定
N = 1000  # 粒子数
L = 10  # 模拟盒子边长
rc = 2.5  # 截断半径
rc2 = rc ** 2  # 预计算截断半径平方
dr = 0.1  # 初始位移步长
n_equilibrium = 200  # 平衡步数（建议至少200，确保体系稳定）
n_production = 100  # 生产步数（建议至少100，减少统计误差）
rho = N / L ** 3  # 数密度
V = L ** 3  # 盒子体积

'''辅助函数'''


def minimum_image_distance(dr, L):
    """最小镜像法修正距离向量（numpy向量化版）"""
    dr = np.array(dr)
    dr = dr - L * np.round(dr / L)
    return dr


def L_J_energy_force(r2):
    """
    计算L-J势能和力（用于维里定理）
    r2: 距离平方
    返回：势能值, 维里项（r·dU/dr）
    """
    r2_inv = 1.0 / r2
    r6_inv = r2_inv ** 3
    r12_inv = r6_inv ** 2

    # L-J势能
    energy = 4 * (r12_inv - r6_inv)
    # 维里项：r·(dU/dr) = 24*(2r^-12 - r^-6) = 24*(2*r12_inv - r6_inv)
    virial = 24 * (2 * r12_inv - r6_inv)

    return energy, virial


def particle_energy(i, positions, L, rc2):
    """计算单个粒子的势能（优化版）"""
    energy = 0.0
    pos_i = positions[i]

    # 批量计算距离 利用广播机制计算所有粒子
    diff = positions - pos_i
    diff = minimum_image_distance(diff, L)
    r2 = np.sum(diff ** 2, axis=1)

    # 筛选有效距离（截断半径内 + 非自身）
    mask = (r2 < rc2) & (r2 > 1e-10)
    r2_valid = r2[mask]

    # 批量计算势能
    if len(r2_valid) > 0:
        r2_inv = 1.0 / r2_valid
        r6_inv = r2_inv ** 3
        r12_inv = r6_inv ** 2
        energy = np.sum(4 * (r12_inv - r6_inv))

    return energy


def calculate_total_energy_virial(positions, L, rc2):
    """
    计算体系总势能和总维里（用于压强计算）
    返回：总势能, 总维里
    """
    total_energy = 0.0
    total_virial = 0.0
    N = len(positions)

    for i in range(N):
        pos_i = positions[i]
        # 只计算i<j，避免重复
        for j in range(i + 1, N):
            diff = positions[j] - pos_i
            diff = minimum_image_distance(diff, L)
            r2 = np.sum(diff ** 2)

            if r2 < rc2 and r2 > 1e-10:
                energy, virial = L_J_energy_force(r2)
                total_energy += energy
                total_virial += virial

    return total_energy, total_virial


def calculate_pressure(total_virial, T, rho, V):
    """
    基于维里定理计算压强（约化单位）
    公式：P = ρT + (1/(3V)) * <总维里>
    """
    pressure = rho * T + (total_virial) / (3 * V)
    return pressure


def apply_pbc(pos, L):
    """应用周期性边界条件（numpy版）"""
    pos = np.array(pos)
    pos = pos % L
    return pos


def initialize_fcc_lattice_np(N, L):
    """初始化FCC晶格（numpy版）"""
    basis = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.5, 0.5],
                      [0.5, 0.0, 0.5],
                      [0.5, 0.5, 0.0]])

    n_cells = int(math.ceil((N / 4) ** (1 / 3)))
    a = L / n_cells

    # 生成晶胞网格
    grid = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                grid.append([ix, iy, iz])
    grid = np.array(grid)

    # 生成所有原子位置
    all_positions = []
    for g in grid:
        for b in basis:
            pos = (g + b) * a
            all_positions.append(pos)

    all_positions = np.array(all_positions)
    np.random.shuffle(all_positions)

    # 确保数量足够
    if len(all_positions) < N:
        extra = np.random.rand(N - len(all_positions), 3) * L
        all_positions = np.vstack([all_positions, extra])

    return all_positions[:N]


def mc_simulation(T, N, L, rc2, dr, n_equilibrium, n_production):
    """
    单温度下的MC模拟主函数
    返回：平均势能（单粒子）, 平均压强
    """
    # 初始化
    positions = initialize_fcc_lattice_np(N, L)


    # 初始总势能
    U_total, _ = calculate_total_energy_virial(positions, L, rc2)

    """平衡阶段"""
    n_accepted_eq = 0
    for step in range(n_equilibrium * N):
        # 随机选粒子
        i = random.randint(0, N - 1)
        old_pos = positions[i].copy()
        old_U_i = particle_energy(i, positions, L, rc2)

        # 随机位移
        dx = dr * (random.random() - 0.5)
        dy = dr * (random.random() - 0.5)
        dz = dr * (random.random() - 0.5)
        new_pos = old_pos + [dx, dy, dz]
        new_pos = apply_pbc(new_pos, L)

        # 新势能
        new_U_i = particle_energy(i, np.vstack([positions[:i], new_pos, positions[i + 1:]]), L, rc2)
        dU = new_U_i - old_U_i

        # Metropolis准则
        if dU <= 0 or random.random() < math.exp(-dU / T):
            positions[i] = new_pos
            U_total += dU
            n_accepted_eq += 1

        # 调整步长
        if (step + 1) % N == 0:
            acceptance_rate = n_accepted_eq / N
            if acceptance_rate < 0.4:
                dr *= 0.9
            elif acceptance_rate > 0.6:
                dr *= 1.1
            n_accepted_eq = 0

    """生产阶段"""
    U_sum = 0.0
    P_sum = 0.0
    n_samples = 0
    n_accepted_pro = 0

    for step in range(n_production * N):
        # 随机选粒子
        i = random.randint(0, N - 1)
        old_pos = positions[i].copy()
        old_U_i = particle_energy(i, positions, L, rc2)

        # 随机位移
        dx = dr * (random.random() - 0.5)
        dy = dr * (random.random() - 0.5)
        dz = dr * (random.random() - 0.5)
        new_pos = old_pos + [dx, dy, dz]
        new_pos = apply_pbc(new_pos, L)

        # 新势能
        new_U_i = particle_energy(i, np.vstack([positions[:i], new_pos, positions[i + 1:]]), L, rc2)
        dU = new_U_i - old_U_i

        # Metropolis准则
        if dU <= 0 or random.random() < math.exp(-dU / T):
            positions[i] = new_pos
            U_total += dU
            n_accepted_pro += 1

        # 每N步采样
        if (step + 1) % N == 0:
            # 调整步长
            acceptance_rate = n_accepted_pro / N
            if acceptance_rate < 0.4:
                dr *= 0.9
            elif acceptance_rate > 0.6:
                dr *= 1.1
            n_accepted_pro = 0

            # 采样势能和压强
            U_sum += U_total
            # 计算当前压强
            _, total_virial = calculate_total_energy_virial(positions, L, rc2)
            P = calculate_pressure(total_virial, T, rho, V)
            P_sum += P
            n_samples += 1

    # 计算平均值
    if n_samples > 0:
        U_avg = U_sum / n_samples
        U_per_particle = U_avg / N
        P_avg = P_sum / n_samples
        return U_per_particle, P_avg
    else:
        return None, None


def run_P_T_simulation(T_list, N, L, rc2, dr, n_equilibrium, n_production):
    """
    多温度下的模拟，获取P-T关系
    T_list: 温度列表
    返回：温度列表, 压强列表, 单粒子势能列表
    """
    P_results = []
    U_results = []

    print("开始P-T关系模拟：")
    for T in T_list:
        print(f"正在模拟温度 T = {T:.2f}...")
        U_per_particle, P_avg = mc_simulation(T, N, L, rc2, dr, n_equilibrium, n_production)

        if U_per_particle is not None and P_avg is not None:
            U_results.append(U_per_particle)
            P_results.append(P_avg)
            print(f"  T={T:.2f}: 单粒子势能={U_per_particle:.4f}, 平均压强={P_avg:.4f}")
        else:
            print(f"  T={T:.2f}: 模拟失败")

    return T_list, P_results, U_results


def plot_P_T_relation(T_list, P_results, U_results):
    """可视化P-T关系（双Y轴）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 压强-T 曲线（主Y轴）
    color = 'tab:red'
    ax1.set_xlabel('温度 T (约化单位)')
    ax1.set_ylabel('压强 P (约化单位)', color=color)
    ax1.plot(T_list, P_results, 'o-', color=color, label='压强')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # 单粒子势能-T 曲线（次Y轴）
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('单粒子势能 U/N (约化单位)', color=color)
    ax2.plot(T_list, U_results, 's-', color=color, label='单粒子势能')
    ax2.tick_params(axis='y', labelcolor=color)

    # 标题和图例
    fig.suptitle('L-J流体 NVT系综 P-T 关系 (ρ=1.0)', fontsize=14)
    fig.tight_layout()  # 调整布局避免标签重叠

    # 保存图片
    plt.savefig('LJ_P_T_relation.png', dpi=300, bbox_inches='tight')
    plt.show()


### 主程序执行
if __name__ == '__main__':
    # 1. 定义温度列表（覆盖液-气相变区间）
    T_list = np.linspace(0.5, 3.0, 10)  # 0.5到3.0，共10个温度点

    # 2. 运行多温度模拟
    T_results, P_results, U_results = run_P_T_simulation(
        T_list=T_list,
        N=N,
        L=L,
        rc2=rc2,
        dr=dr,
        n_equilibrium=n_equilibrium,
        n_production=n_production
    )

    # 3. 可视化结果
    if len(P_results) > 0 and len(U_results) > 0:
        plot_P_T_relation(T_results, P_results, U_results)

        # 输出数据表格
        print("\n=== 模拟结果汇总 ===")
        print(f"{'温度T':<8} {'压强P':<10} {'单粒子势能U/N':<15}")
        print("-" * 35)
        for T, P, U in zip(T_results, P_results, U_results):
            print(f"{T:<8.2f} {P:<10.4f} {U:<15.4f}")
    else:
        print("模拟未获取到有效数据！")
