# 双 IMU 手势分类 MLP

本目录使用双 IMU 动作窗口的 84 维统计特征识别 8 类手势。下文记录的 50 轮实验在 16 条真实测试录制上识别正确 14 条（87.50%）。两次误判分别来自抓取和释放。

## 代码与数据接口

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

## 训练与测试

在本机 Anaconda Prompt 中激活环境，进入本目录后执行：

```bat
conda activate py312_pytorch214
cd /d "D:\Engineering Software\PyProjects\5205_final_proj\Python\deep_learning"
python train_test_mlp.py
```

默认使用 Adam、学习率 0.001、批大小 16、随机种子 42，训练 50 轮。有 CUDA 时使用 GPU。交叉熵的类别权重按训练集计算，类别 c 的权重为 `N / (8 × n_c)`，其中 N 为训练样本总数，n_c 为该类训练样本数。可用 `--epochs`、`--batch-size`、`--lr` 和 `--device` 调整运行参数。

控制台逐轮显示训练损失、验证损失和验证准确率，最后打印测试指标、逐类报告和混淆矩阵。每次运行在 `runs/` 下生成带时间戳的检查点，包含模型权重及标准化参数、类别顺序、最佳轮次、最低验证损失、测试准确率、训练配置以及数据集版本和校验清单摘要。逐轮日志及完整测试报告仅输出到控制台。

## 已记录的实验结果
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

## 导出到 Arduino

在本目录执行以下命令可导出上表对应的模型。导出其他训练结果时，将参数换成控制台打印的实际检查点路径。

```bat
python export_arduino.py runs/gesture_mlp_20261104_215212_312180.pt
```

将[生成的 gesture_mlp.h](arduino_export/gesture_mlp_20261104_215212_312180/gesture_mlp.h) 放入 Arduino 草图目录并包含该头文件。调用 `gesture_mlp::predict(features, logits)`，传入按训练顺序排列的 84 个未标准化浮点特征和长度为 8 的输出数组。返回值是类别编号，可用 `gesture_mlp::class_name(id)` 获取名称。Mega 2560 通过 `PROGMEM` 从 Flash 读取参数。实时传感器信号的分窗与特征提取需由调用方提供。

电脑端已重新读取该头文件中的参数，按导出的运算顺序与 PyTorch 对照。16 条测试输入的预测类别全部一致，logit 最大绝对误差约为 9.54 × 10⁻⁷。此结果验证了参数导出的数值一致性，Arduino 编译和开发板实测仍待完成。
