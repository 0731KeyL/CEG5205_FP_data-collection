# 双 IMU 手势分类 MLP

本目录使用双 IMU 动作窗口的 84 维统计特征进行手势分类，当前默认采用六类 dataset_v004。原八类代码与结果的演进记录见下文。

## 六类版本更新（2026-11-05）

当前任务由 8 类调整为 6 类，原 `grasp`（抓取）和 `release`（释放）统一归入 `other`。标签编号依次为 idle=0、forward=1、backward=2、left=3、right=4、other=5。

| 数据集 | 用途                        |
| --- |---------------------------|
| `dataset_v002` | 保留原八类数据。                  |
| `dataset_v003` | 合并标签后的六类数据，沿用原划分。         |
| `dataset_v004` | 当前训练脚本默认数据集。v003基础上重新打乱划分 |

| 类别 | 训练真实 | 训练增强 | 训练合计 | 验证 | 测试 |
| --- | ---: | ---: | ---: | ---: | ---: |
| idle | 6 | 18 | 24 | 2 | 2 |
| forward | 6 | 18 | 24 | 2 | 2 |
| backward | 6 | 18 | 24 | 2 | 2 |
| left | 6 | 18 | 24 | 2 | 2 |
| right | 6 | 18 | 24 | 2 | 2 |
| other | 15 | 45 | 60 | 3 | 6 |
| 合计 | 45 | 135 | 180 | 13 | 16 |

### 网络与 Dropout

当前网络为 `84 → Linear(84,16) → ReLU → Dropout → Linear(16,6)`，共有 1,462 个可训练参数。Dropout 放在隐藏层 ReLU 之后、第二个全连接层之前。代码已预留 Dropout 概率配置，目前为 `nn.Dropout(p=0.0)`，即关闭。需要启用时可修改为 `p=0.1`。当前开关通过代码中的概率值控制。

模型根据 `num_classes` 设置输出数，训练评估和 Arduino 导出根据数据集或检查点中的类别数处理。当前标准化流程仍在模型前向传播中执行。

### 可视化程序

[visualize_error_vs_idle.py](visualize_error_vs_idle.py) 比较 v003 的误判 release 样本与一条真实 idle 测试样本，展示 12 通道窗口、84 维标准化特征、最大差异特征及模型预测概率。直接运行可显示图形，添加 `--no-show` 可仅保存图片。默认输出位于 `visualizations/error_vs_idle/`，并保留针对 v003 和对应检查点的默认设置。

### 六类训练记录

据本次实验记录，已重新进行三次六类训练。以下详细结果对应第一次训练提供的控制台输出，其余两次的详细指标未在本节列出。

检查点为 [gesture_mlp_6class_20261105_202851_523541.pt](runs/gesture_mlp_6class_20261105_202851_523541.pt)。已核对其中的 dataset_id 为 `dataset_v004_20261105T121912Z_sixclass`，最佳轮次和测试准确率与控制台记录一致。使用 CUDA、Adam、学习率 0.001、批大小 16、随机种子 42，共训练 50 轮，Dropout 关闭。

| 指标 | 第一次六类训练结果 |
| --- | ---: |
| 最佳轮次 | 50 |
| 第 50 轮训练损失 | 0.0105 |
| 第 50 轮验证损失 | 0.0515 |
| 验证准确率 | 100%（13/13），第 7 轮起达到 100% |
| 测试损失 | 0.0641 |
| 测试准确率 | 100%（16/16） |
| 宏平均 Precision / Recall / F1 | 1.000 / 1.000 / 1.000 |
| 加权平均 Precision / Recall / F1 | 1.000 / 1.000 / 1.000 |

六类的 Precision、Recall 和 F1 均为 1.000。混淆矩阵按 idle、forward、backward、left、right、other 排列，行为真实类别、列为预测类别。

```text
[[2 0 0 0 0 0]
 [0 2 0 0 0 0]
 [0 0 2 0 0 0]
 [0 0 0 2 0 0]
 [0 0 0 0 2 0]
 [0 0 0 0 0 6]]
```

使用当前版本可直接执行 `python train_test_mlp.py`。导出本次六类模型时执行：

```bat
python export_arduino.py runs/gesture_mlp_6class_20261105_202851_523541.pt
```

下文八类数据和实验结果保留为历史记录，适用于原 dataset_v002。

## 历史八类代码与数据接口

| 文件 | 功能 |
| --- | --- |
| [gesture_mlp.py](gesture_mlp.py) | 定义 `GestureMLP`。输入形状为 `(batch, 84)`，输出为 `(batch, 8)` 的 logits。 |
| [train_test_mlp.py](train_test_mlp.py) | 校验数据包，构建训练批次，逐轮训练和验证。训练结束后恢复验证损失最低的权重，测试一次并保存检查点。 |
| [export_arduino.py](export_arduino.py) | 将选定的检查点导出为包含标准化参数、权重、类别名和前向推理函数的 Arduino 头文件。 |

数据来自 [dataset_v002](../dataset_v002/)。训练脚本通过 [load_split](../load_dataset.py) 读取各划分的 `features` 和 `y`。每条录制对应约 1 秒、50 × 12 的信号窗口。数据包已提取 84 维特征。输入模型时保留原始特征值，由模型使用 `normalization.npz` 中的 `feature_mean` 和 `feature_std` 完成一次标准化。

| 划分 | 真实录制 | 增强样本 | 输入模型的样本数 |
| --- | ---: | ---: | ---: |
| 训练 | 45 | 135 | 180 |
| 验证 | 13 | 0 | 13 |
| 测试 | 16 | 0 | 16 |

网络为 `Linear(84, 16) → ReLU → Linear(16, 8)`，共有 1,496 个可训练参数。预测取最大 logit 的索引。索引 0 至 7 依次对应 `idle, forward, backward, left, right, grasp, release, other`，见[标签映射](../dataset_v002/label_map.json)。

## 环境配置

已在 Windows 本机的 Conda 环境 `py312_pytorch214` 中运行训练、测试和模型导出。以下为本机已安装版本，供复现时核对。

| 组件 | 版本或配置 |
| --- | --- |
| Python | 3.12.16 |
| PyTorch | 2.14.1+cu126 |
| PyTorch CUDA 运行时 | 12.6 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，约 8 GB 显存 |
| NVIDIA 驱动 | 566.24 |

本目录直接依赖 PyTorch、NumPy 和 scikit-learn。scikit-learn 用于测试分类报告及混淆矩阵，导出脚本依赖 PyTorch。标准库模块随 Python 提供。当前环境已补齐项目依赖，可直接激活使用。

```bat
conda activate py312_pytorch214
python -m pip check
python -c "import sys, torch, numpy, sklearn; print(sys.executable); print(torch.__version__); print(numpy.__version__); print(sklearn.__version__); print('CUDA available:', torch.cuda.is_available())"
```

本机解释器路径为 `D:\Engineering_Software_nospace\Anaconda3\envs\py312_pytorch214\python.exe`。在 PyCharm 的项目 Python Interpreter 设置中选择这一解释器，然后运行本目录脚本。脚本通过自身位置定位默认数据集，工作目录可使用项目根目录或本目录。

配置新环境时，先创建并激活 Python 3.12 环境，再安装适配目标机器的 PyTorch 构建，以及 NumPy 和 scikit-learn。下列命令提供基础安装方式，pip 会解析版本约束下的依赖。CUDA 可用性需用上面的检查命令确认，训练脚本也支持通过 `--device cpu` 运行。

```bat
conda create -n ceg5205_mlp python=3.12
conda activate ceg5205_mlp
python -m pip install torch numpy scikit-learn
python -m pip check
```

上面的基础安装命令未锁定完整依赖，可能与记录的实验环境版本不同。若需要复现本机完整环境，可在源电脑激活 `py312_pytorch214` 后执行 `conda env export --no-builds > environment.yml`，在目标电脑使用 `conda env create -f environment.yml`。迁移前应删除导出文件中的本机 `prefix` 行，并核对 pip 源是否提供对应的 PyTorch CUDA 构建。

整个项目的采集、分析和可视化还使用 pandas 3.1.0、matplotlib 3.11.2、pyserial 3.5、pyqtgraph 0.14.0 和 PyQt5 5.15.11。它们已安装在当前本机环境中，项目依赖清单见 [requirements.txt](../requirements.txt)。该清单未包含 PyTorch，且未锁定版本。配置已有环境时，应先检查已安装包，再补齐缺失依赖，保留经过验证的现有版本。

## 历史八类训练与测试（当前默认已改为 v004）

在本机 Anaconda Prompt 中激活环境，进入本目录后执行：

```bat
conda activate py312_pytorch214
cd /d "D:\Engineering Software\PyProjects\5205_final_proj\Python\deep_learning"
python train_test_mlp.py
```

默认使用 Adam、学习率 0.001、批大小 16、随机种子 42，训练 50 轮。有 CUDA 时使用 GPU。交叉熵的类别权重按训练集计算，类别 c 的权重为 `N / (8 × n_c)`，其中 N 为训练样本总数，n_c 为该类训练样本数。可用 `--epochs`、`--batch-size`、`--lr` 和 `--device` 调整运行参数。

控制台逐轮显示训练损失、验证损失和验证准确率，最后打印测试指标、逐类报告和混淆矩阵。每次运行在 `runs/` 下生成带时间戳的检查点，包含模型权重及标准化参数、类别顺序、最佳轮次、最低验证损失、测试准确率、训练配置以及数据集版本和校验清单摘要。逐轮日志及完整测试报告仅输出到控制台。

## 历史八类实验结果
共进行三次训练，结果保持一致
以下数字对应gesture_mlp_20261104_220033_050334.pt本次训练结果
以下为其控制台输出，使用 CUDA 和上述默认配置。

| 指标 | 结果 |
| --- | ---: |
| 验证损失最低轮次 | 50 |
| 第 50 轮训练损失 / 验证损失 | 0.0159 / 0.0716 |
| 验证准确率 | 13/13（100%） |
| 测试损失 / 准确率 | 0.7542 / 14/16（87.50%） |
| 测试宏平均 F1 / 加权平均 F1 | 0.711 / 0.822 |

测试中，唯一的 `grasp` 被判为 `other`，唯一的 `release` 被判为 `idle`。其余六类的 14 条真实录制全部判对。抓取与释放各只有 1 条真实训练录制，验证集也均无这两类样本，因此验证准确率只反映六个有样本类别的表现。所有录制来自同一名被试的左手，跨被试识别效果仍待验证。

## 历史八类 Arduino 导出

在本目录执行以下命令可导出上表对应的模型。导出其他训练结果时，将参数换成控制台打印的实际检查点路径。

```bat
python export_arduino.py runs/gesture_mlp_20261104_215212_312180.pt
```

将[生成的 gesture_mlp.h](arduino_export/gesture_mlp_20261104_215212_312180/gesture_mlp.h) 放入 Arduino 草图目录并包含该头文件。调用 `gesture_mlp::predict(features, logits)`，传入按训练顺序排列的 84 个未标准化浮点特征和长度为 8 的输出数组。返回值是类别编号，可用 `gesture_mlp::class_name(id)` 获取名称。Mega 2560 通过 `PROGMEM` 从 Flash 读取参数。实时传感器信号的分窗与特征提取需由调用方提供。

电脑端已重新读取该头文件中的参数，按导出的运算顺序与 PyTorch 对照。16 条测试输入的预测类别全部一致，logit 最大绝对误差约为 9.54 × 10⁻⁷。此结果验证了参数导出的数值一致性，Arduino 编译和开发板实测仍待完成。
