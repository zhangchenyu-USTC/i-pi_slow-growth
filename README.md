# i-PI-3.3.0-slow-growth

本项目在 i-PI 3.3.0 的基础上实现了严格约束的 slow-growth 分子动力学。

i-PI 可以通过 PLUMED 的 Python 接口使用多种增强采样方法。但是，与严格约束的 slow-growth 不同，PLUMED 的
[`MOVINGRESTRAINT`](https://www.plumed.org/doc-v2.10/user-doc/html/_m_o_v_i_n_g_r_e_s_t_r_a_i_n_t.html)
采用移动谐振弹簧牵引 CV，允许实际 CV 偏离目标，弹簧系数、时间步和拉动速率都会影响有限采样下的结果。

本项目通过 SHAKE 算法，在每个完成的 MD 步将 CV 限制到目标值的数值容差内，
不需要选择弹簧系数；也提供带移动约束速度投影的 RATTLE 模式。
严格约束由 i-PI 执行，能量和物理力仍由普通 i-PI 客户端提供，例如 VASP、CP2K、MACE。

## 安装

需要 Python ≥ 3.10 和 NumPy < 3。建议使用独立虚拟环境，避免覆盖其他 i-PI 分支。

在本仓库根目录安装：

```bash
python -m pip install -e .
```

## 输入示例

在常规 i-PI `input.xml` 的 `<system>` 中，用以下模块替换原来的 `<motion>`。
初始化、力客户端、输出、总步数等仍按普通 i-PI 设置；NVT 温度在 `<ensemble>` 中设置。

完整的 VASP 客户端输入与初始结构见 [Volmer 示例](examples/features/slow_growth/volmer/README.md)。
该示例改编自 Qin、Vegge 和 Hansen 的
[*Cation-Coordinated Inner-Sphere CO2 Electroreduction at Au–Water Interfaces*](https://doi.org/10.1021/jacs.2c11643)
及其 [开放数据](https://doi.org/10.5281/zenodo.7474433)中的 K⁺ 配位水分子 Volmer 路径。
反应坐标为 $\xi=d(\mathrm O,\mathrm H)-d(\mathrm H,\mathrm{Au})$，对应下方的原子编号与系数。
本例使用 Γ 点、i-PI SHAKE 和 SVR（τ=200 fs，容差 1e−6 Å）。

```xml
    <motion mode="slow_growth">
      <fixcom>false</fixcom>
      <slow_growth mode="nvt" algorithm="shake">
        <timestep units="femtosecond">1.0</timestep>
        <thermostat mode="svr">
          <tau units="femtosecond">200.0</tau>
        </thermostat>
        <distance>
          <atoms>[119, 68, 68, 134]</atoms>
          <coefficients>[1.0, -1.0]</coefficients>
          <start units="angstrom">-1.2413615592403282</start>
          <increment units="angstrom">0.001</increment>
        </distance>
        <shake>
          <tolerance units="angstrom">1.0e-6</tolerance>
          <max_iterations>1000</max_iterations>
        </shake>
      </slow_growth>
    </motion>
```

原子编号为 **0-based**。初始结构的 CV 必须与 `start` 相符，否则报错。
第一步使用 `start`，完成后才增加 `increment`；负增量反向扫描，零增量固定 CV，可用于 Blue-Moon 采样。
将 `algorithm="shake"` 改为 `"rattle"`，可同时显式约束 CV 的变化速度。

支持标量长度型 CV：键长、键长线性组合（包括键长差）、原子到两原子几何中点
的距离，以及笛卡尔坐标分量线性组合（例如两个原子的 Z 坐标差）。
当前只支持单 bead、固定晶胞、NVE 或 NVT（CSVR/SVR）、一个 force/MTS 层。


## 自由能分析

### 1. 在 properties 中输出什么？

可以在 `<simulation>` 的 `<output>` 中加入以下输出；文件名和 stride 可按需要调整：

```xml
<properties stride="1" filename="out">
  [ step, time{femtosecond}, temperature{kelvin},
    slow_growth_cv{angstrom}, slow_growth_target{angstrom},
    slow_growth_error{angstrom}, slow_growth_gradient{ev/ang},
    slow_growth_metric, slow_growth_gkt{ev/ang},
    slow_growth_work{electronvolt} ]
</properties>
```

| 输出 | 含义 | 自由能分析中的用途 |
| --- | --- | --- |
| `slow_growth_cv` | 当前实际 CV，记为 ξ | 必需：分箱和积分的反应坐标 |
| `slow_growth_gradient` | 原始瞬时梯度 g_raw = -λ_internal | 必需：平均力估计；不能对它取绝对值 |
| `slow_growth_metric` | Blue-Moon 权重 w = Z^(-1/2) | 一般 CV 的加权平均必需；常量权重可约去 |
| `slow_growth_gkt` | 几何/质量度量修正 G k_B T | 一般 CV 的修正必需；已知为零的情况可省略修正 |
| `slow_growth_target`, `slow_growth_error` | 本步目标、实际 CV 与目标之差 | 检查约束是否满足|
| `slow_growth_work` | 原始梯度的累计积分 | 仅作在线观察，不是通用后处理的必需列 |

还可输出总/位置/速度约束乘子、Z、加权梯度、SHAKE 迭代数和速度残差，详见
[使用说明](docs/slow-growth-usage.md#3-建议输出)。`slow_growth_lambda` 与原始梯度符号相反，
不需要同时输出两者才能积分。

### 2. 如何获得自由能曲线？

对 NVT 数据，先检查轨迹和温度，按真实续算顺序拼接文件、去掉重启重复帧。
连续 slow-growth 轨迹可按 CV 分箱（例如先试 0.05 Å）；固定 CV 窗口则先丢弃平衡段，
再对每个窗口的生产样本求平均。

每箱/窗口的 Blue-Moon 平均梯度为：

$\bar g_j=\frac{\sum_{n\in j}w_n[g_{\mathrm{raw},n}+(Gk_BT)_n]}{\sum_{n\in j}w_n}$。

随后以箱平均 CV 或窗口目标值为横坐标，用梯形积分：

$A_{j+1}=A_j+\frac{\bar g_j+\bar g_{j+1}}{2}(\xi_{j+1}-\xi_j)$。

将反应物参考点设为零点，即得到沿所选 CV 的相对自由能曲线。上面的梯度和 GkT 已输出为
eV/Å，CV 为 Å，因此积分单位为 eV。若反应物、过渡区和产物盆地已经识别，
可读取沿该 CV 的活化自由能 $\Delta A^\ddagger=A_{\mathrm{TS}}-A_{\mathrm R}$
和反应自由能 $\Delta A=A_{\mathrm P}-A_{\mathrm R}$。

注意，只有足够慢的扫描或充分平衡的固定窗口，平均梯度积分才可作为平衡自由能的可靠估计。

### 3. slow_growth_work 为什么不能直接代替上述分析？

`slow_growth_work` 只对原始梯度 g_raw 沿实际 CV 做逐步梯形积分，未加入 GkT、也不执行 Blue-Moon 加权平均。

单一单位系数键长、当前 midpoint 距离和笛卡尔坐标线性组合的 Z 为常数、G=0，
Blue-Moon 修正退化，原始梯度可以直接平均后积分。一般键长组合中 Z 可能随构型变化，
应使用上述加权公式，不能直接把 `slow_growth_work` 当作已修正的自由能。
固定窗口的 `increment=0` 时，这个累计量基本不增长，也不代表不同窗口之间的自由能差。

### 4. 自由能结果需要做增长速率收敛测试

单条 slow-growth 轨迹给出的是**当前增长速率下的单条自由能估计曲线**，
不能代表收敛的平衡自由能。增长速率为 `increment / timestep`；
固定时间步并逐步减小增量，可增加体系经过同一 CV 区间时的采样时间。
比较时应保持扫描的 CV 范围一致，并相应增加总步数。

自由能结果的收敛性，需要比较不同增长速率下的平均梯度、能垒和反应自由能，并结合独立重复轨迹及正反向扫描进行检查。
即使 CV 严格受限，水网络、离子配位等其他自由度仍可能来不及平衡。
采样收敛问题可参考
[*Constant-Potential Machine Learning Force Field for the Electrochemical Interface*](https://doi.org/10.1021/acs.jctc.5c00784)，
J. Chem. Theory Comput. (2025)；合适的增长速率需要针对当前体系验证。

### 5. 推荐的计算方法

也可以先用较快的增长速率扫描整个 CV 范围，保存不同 CV 处的结构，
再分别建立固定窗口，把每个窗口的 `start` 设置为其初始结构的实际 CV，
并将 `increment` 设为 `0.0`进行blue-moon模拟。

固定窗口建议使用 `algorithm="rattle"`，同时满足位置约束和零 CV 变化速度条件。
每个窗口先平衡，再进行足够长的生产采样；检查分段平均力、不同初始构型以及误差随采样长度是否稳定，
最后用第 2 点的公式对各窗口平均梯度积分。
建议用新的窗口输入重新初始化；不能只修改旧 checkpoint 的 `increment`，
因为其中还保存着下一步目标和原扫描的积分历史。


这种做法便于单独延长收敛较慢的窗口，可能获得更好的收敛性，但仍需检查其他自由度是否充分采样。

## 许可与引用

本项目基于 i-PI 3.3.0 开发，保留其 [许可证](licenses/LICENSE.md) 和原始版权声明。
使用 i-PI 请引用 [i-PI 3.0 论文](https://doi.org/10.1063/5.0215869)。
