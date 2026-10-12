# Volmer：K⁺ 配位水分子解离

## 来源与改编

本例改编自 Xueping Qin、Tejs Vegge 和 Heine Anton Hansen 的
[*Cation-Coordinated Inner-Sphere CO2 Electroreduction at Au–Water Interfaces*](https://doi.org/10.1021/jacs.2c11643)，
J. Am. Chem. Soc. **145**, 1897–1905 (2023)。
结构与反应路径来自其 [Zenodo 开放数据](https://doi.org/10.5281/zenodo.7474433)，
`SG_HER.zip` 中的 `SG_HER/Volmer/K_coordinated_H2O`。

结构数据沿用原数据的 [CC BY 4.0 许可](https://creativecommons.org/licenses/by/4.0/)，
归属上述数据作者；本仓库的代码许可证不替代数据的原许可。
**POTCAR、VASP 程序、运行结果和原服务器专用脚本不随本例分发。**

## 反应坐标与设置

反应坐标为 $\xi=d(\mathrm O,\mathrm H)-d(\mathrm H,\mathrm{Au})$。
i-PI 的 0-based 编号为 O=119、H=68、Au=134，对应：

```xml
<distance>
  <atoms>[119, 68, 68, 134]</atoms>
  <coefficients>[1.0, -1.0]</coefficients>
  <start units="angstrom">-1.2413615592403282</start>
  <increment units="angstrom">0.001</increment>
</distance>
```

| 参数 | 本例设置 |
| --- | --- |
| 温度与系综 | 298 K，NVT |
| 算法、时间步 | SHAKE，1 fs |
| SVR 时间常数 | 200 fs |
| CV 容差 | 1e−6 Å |
| 扫描增量、总步数 | +0.001 Å/步，2000 步 |
| 最后一帧目标 | 0.7576384407596719 Å |
| 结构、固定原子 | 173 原子；固定 24 个 Au，编号见 input.xml |
| VASP 力设置 | ENCUT=400 eV，IVDW=11，ALGO=V，Γ-only 1×1×1 |

## 运行

参考sbatch.slurm文件。
