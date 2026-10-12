# 严格约束 slow-growth 使用说明

本文说明当前实现的输入、支持范围、输出和自由能后处理。

## 1. 模式与参数

在 `<motion mode="slow_growth">` 中设置 `<slow_growth>`，完整模块见根目录 README。
初次运行只需用户参数；`current_target`、`last_target`、`previous_*`、`last_*`、
`completed_steps`、`cumulative_work` 和 midpoint 参考值等由 checkpoint 自动保存。

| 参数 | 用途与约定 |
| --- | --- |
| `<slow_growth mode="nve\|nvt">` | 经典 NVE 或 NVT；默认 NVE |
| `algorithm="shake\|rattle"` | 默认 SHAKE 位置约束；RATTLE 额外投影速度，使 CV 变化速度匹配扫描速度 |
| `<timestep units="femtosecond">` | 完整 MD 时间步；应显式给值和单位 |
| `<thermostat mode="svr"><tau ...>` | NVT 仅支持 i-PI 的 CSVR/SVR；温度在 `<ensemble>` 设置 |
| `<distance coordinate_type="...">` | CV 类型：`linear`（默认）、`midpoint`、`cartesian` |
| `<atoms>` | 0-based 原子编号；具体排列由 CV 类型决定 |
| `<coefficients>` | 距离或坐标分量各项的系数；省略时每项为 1；midpoint 不接受此参数 |
| `<components>` | 仅 Cartesian 使用，逐项指定 `0=x`、`1=y`、`2=z` |
| `<start units="angstrom">` | 首个完成 MD 步的目标；初始结构必须在 tolerance 内匹配 |
| `<increment units="angstrom">` | 每个完整步之后增加的目标值；可以为正、负或 0 |
| `<shake><tolerance units="angstrom">` | 允许的 CV 绝对误差；示例建议显式设置，不能把无单位默认值当作 Å |
| `<shake><max_iterations>` | 单步位置投影的最大迭代次数；默认 1000；不收敛立即报错 |

`fixcom`、`fixatoms_dof` 是上游 motion 参数，不是本扩展新增参数。示例显式设置
`<fixcom>false</fixcom>`，尤其对绝对坐标 CV，不宜额外移除与约束冲突的质心运动。
固定自由度不得与 CV 的非零梯度自由度冲突；距离和 midpoint 模式拒绝固定参与原子的任何坐标分量。

设总共完成 N 步，则最后一帧使用的目标为 `start + (N - 1) * increment`。
若要求最后一帧恰好到 `end`，使用 `increment = (end - start) / (N - 1)`；
扫描速率为 `increment / timestep`。重启时应使用 checkpoint，不应重新从原始 input 初始化。

## 2. CV 输入

以下片段都放在 `<slow_growth>` 内。`start` 只是写法示范，必须替换为实际初始结构的 CV。

### 2.1 单键长：ξ = d(0, 1)

```xml
<distance coordinate_type="linear">
  <atoms>[0, 1]</atoms>
  <start units="angstrom">1.0</start>
  <increment units="angstrom">0.001</increment>
</distance>
```

距离使用按晶胞分数坐标选择周期像的 minimum-image 约定；需要有限、非奇异晶胞。
单一单位系数距离的目标必须大于 0。实际路径不要经过距离为零或周期像切换的不可微位置。

### 2.2 键长差：ξ = d(0, 1) - d(1, 2)

```xml
<distance coordinate_type="linear">
  <atoms>[0, 1, 1, 2]</atoms>
  <coefficients>[1.0, -1.0]</coefficients>
  <start units="angstrom">-1.0</start>
  <increment units="angstrom">0.001</increment>
</distance>
```

`atoms` 为展平原子对；一个 coefficient 对应一对。一般距离线性组合允许有符号目标。
同一原子出现在多项时，其 CV 梯度会累加；质量度量可能随构型变化，需保留 Blue-Moon 输出。

### 2.3 原子到几何中点的距离

下面是原子 2 到原子 0、1 几何中点的距离；不是质量加权中点。

```xml
<distance coordinate_type="midpoint">
  <atoms>[2, 0, 1]</atoms>
  <start units="angstrom">2.5</start>
  <increment units="angstrom">0.001</increment>
</distance>
```

原子顺序为 `[anchor, first, second]`，三者必须不同，不设置 coefficients。
初始几何由 pairwise minimum image 构造；随后保存连续 midpoint 周期像参考，
避免最终向量跨越半晶胞时任意折回。参考只在位置投影接受后提交，并写入 checkpoint。
它不是每一帧重新寻找全局最短 midpoint 距离；大位移或其他底层周期像分支切换仍需用户检查。
若从扫描结构建立固定窗口，XYZ 快照可能不足以恢复原扫描的周期像分支；
应保留并转移对应 checkpoint 的 `midpoint_reference` 与 `has_midpoint_reference`，
同时按新窗口重新设置目标、步数和积分历史。不要将旧扫描 checkpoint 直接当成新窗口的重启文件。

### 2.4 两原子的 Z 坐标差：ξ = z(0) - z(1)

```xml
<distance coordinate_type="cartesian">
  <atoms>[0, 1]</atoms>
  <components>[2, 2]</components>
  <coefficients>[1.0, -1.0]</coefficients>
  <start units="angstrom">5.0</start>
  <increment units="angstrom">-0.001</increment>
</distance>
```

Cartesian 模式逐项组合 `atoms`、`components` 和 `coefficients`。例如单原子 Z 坐标
使用 `atoms=[0]`、`components=[2]`、`coefficients=[1.0]`。
使用 i-PI 原始笛卡尔坐标，不做 minimum image 或 wrap，也不会从任意折回的结构自动
推断历史非折叠坐标。请准备所需初始周期像，并保持坐标表示连续。

## 3. 建议输出

在 `<simulation><output>` 中加入以下 properties，具体文件名与 stride 可调整：

```xml
<properties stride="1" filename="out">
  [ step, time{femtosecond}, temperature{kelvin},
    slow_growth_cv{angstrom}, slow_growth_target{angstrom},
    slow_growth_error{angstrom}, slow_growth_gradient{ev/ang},
    slow_growth_lambda{ev/ang}, slow_growth_shake_lambda{ev/ang},
    slow_growth_velocity_lambda{ev/ang},
    slow_growth_metric_z, slow_growth_metric, slow_growth_gkt{ev/ang},
    slow_growth_weighted_gradient, slow_growth_work{electronvolt},
    slow_growth_iterations, slow_growth_velocity_error{m/s},
    slow_growth_completed_steps ]
</properties>
```

| 输出 | 含义 |
| --- | --- |
| `slow_growth_cv`, `slow_growth_target`, `slow_growth_error` | 实际 CV、本步使用的目标、两者之差 |
| `slow_growth_lambda` | 内部符号约定的总约束乘子 |
| `slow_growth_shake_lambda`, `slow_growth_velocity_lambda` | 位置乘子与速度投影的等效乘子；SHAKE 模式速度项为 0 |
| `slow_growth_gradient` | 原始瞬时梯度 g_raw = -lambda_internal |
| `slow_growth_metric_z` | 质量度量 Z = J M^(-1) J^T，内部单位 |
| `slow_growth_metric` | Blue-Moon 权重 w = Z^(-1/2)，内部单位 |
| `slow_growth_gkt` | 度量导数修正 G k_B T；一般距离组合可能非零 |
| `slow_growth_weighted_gradient` | w(g_raw + G k_B T)，内部单位 |
| `slow_growth_work` | 沿实际 CV 对原始梯度做梯形积分的累计量，不是 PLUMED 的弹簧协议功 |
| `slow_growth_iterations`, `slow_growth_velocity_error` | 位置迭代数、RATTLE 非齐次速度条件残差 |
| `slow_growth_completed_steps` | 已完成的 slow-growth MD 步数 |

这里 $J=\nabla_q\xi$，约束力为 $F_c=-J^T\lambda_{\mathrm{internal}}$。
SHAKE 乘子来自位置修正；RATTLE 模式还累计速度投影冲量除以时间步。

## 4. 从平均梯度到自由能

在目标温度下充分平衡的 NVT 固定窗口（increment=0），或已验证准静态的 NVT 慢扫描中，按窗口/小 CV 箱计算：

$\bar g_j=\frac{\sum_{n\in j}w_n[g_{\mathrm{raw},n}+(Gk_BT)_n]}{\sum_{n\in j}w_n}$。

随后对 $\bar g_j$ 沿 CV 用梯形积分，给反应物参考点设置零点。
使用上面的 `gradient{ev/ang}` 和 `gkt{ev/ang}` 列进行加权，积分后单位就是 eV。
如果使用内部单位列，应先分别求和再相除，即 `sum(weighted_gradient) / sum(metric)`，
然后把 Hartree/Bohr 转为 eV/Å；不能先逐帧相除再普通平均，否则会抵消权重。
NVE 模式可用于算法和力学检验，不应无条件套用这个正则系综自由能公式。

当前 `slow_growth_work` 只积分原始梯度，不执行上述加权平均。
单一单位系数距离、当前 midpoint 和 Cartesian 线性组合的 Z 为常数、G=0，修正退化；
一般距离组合的 G 通过参与原子坐标上 1e-4 Å 的中心差分计算。
无论 CV 是否严格受限，都应检查采样长度、扫描速率、正反向滞后与重复轨迹。
瞬时梯度、原始累计积分或有限速率曲线不能直接当作已收敛的平衡自由能。

## 5. 运行、重启与限制

使用 VASP 作为力客户端的完整 Volmer 输入与初始结构见
[Volmer 示例](../examples/features/slow_growth/volmer/README.md)。
该示例需要支持 i-PI socket 的 VASP、ASE，以及用户自行提供的有许可 POTCAR；
不属于 GitHub CI 中执行的 DFT 测试。

复制示例到独立运行目录，按实际环境修改 `sbatch.slurm` 中的程序路径，
并准备 POTCAR 后提交。中断后用该运行目录自己的 checkpoint 续算：

```bash
i-pi volmer.restart
```

随后重新启动 VASP 客户端。延长已完成的模拟时，先修改 checkpoint 的 `total_steps`。
当前 GitHub CI 仅运行 slow-growth 专项单元测试，不提交 Slurm 任务或调用真实 DFT。

其他客户端使用常规 i-PI 启动方法。DFT/ML 力由客户端提供，约束几何迭代不调用客户端；
标准路径通常需要初始力加每个 MD 步一次新位置的力。客户端必须有足够的请求寿命，
其 NSW/终止行为取决于实际 driver 版本，不应仅凭 VASP 版本号推断。

仅支持一个标量 CV、nbeads=1、固定晶胞、一个 force/MTS 层、NVE/NVT（SVR）。
不支持配位数、角度、二面角、多独立约束、非线性时间表、NPT/NST 或 PIMD。
现有 PLUMED bias 接口可用于附加势，但本文不提供墙势模块；有额外势时得到的是包含该势的体系自由能。
本扩展没有增加恒电势电子自由度或修改 socket 协议。
