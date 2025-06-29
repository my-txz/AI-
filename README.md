# AI五子棋 - 智能对弈系统
# 问题反馈 - https://github.com/my-txz/AI-/issues/
## 许可
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.0+-green.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

AI五子棋是一个基于Python和Pygame开发的智能五子棋对弈系统，融合了多种先进的人工智能算法，提供了从入门到专业级别的不同难度模式，以及AI对AI的自动对弈功能。

## 项目亮点

### 🧠 多算法融合决策
- **混合智能系统**：结合蒙特卡洛树搜索(MCTS)、模拟退火、PVS搜索等多种算法
- **威胁检测引擎**：精准识别活三、活四、冲四、双三等关键威胁模式
- **自适应策略**：根据局面复杂度动态调整算法深度和资源分配

### 🚀 多级难度系统
| 难度 | 算法特点 | 适合人群 |
|------|----------|----------|
| **Easy** | 启发式优先级算法 | 初学者 |
| **Normal** | 模拟退火+启发式评估 | 进阶玩家 |
| **Difficult** | MCTS+模拟退火+启发式 | 高级玩家 |
| **Hell** | 多进程并行算法融合 | 专业挑战 |
| **Professional** | MCTS+PVS深度搜索 | 大师级别 |

### ⚡ 性能优化
- **多线程/多进程并行**：充分利用多核CPU资源
- **动态资源分配**：根据局面复杂度调整计算强度
- **启发式剪枝**：三阶启发式排序表提升搜索效率
- **时间管理**：智能超时处理保障流畅体验

### 🎮 丰富功能
- **AI对战**：挑战不同难度级别的AI
- **AI自动对弈**：观看AI之间的高水平对决
- **实时分析**：显示威胁分布和AI思考过程
- **响应式UI**：直观的棋盘和操作界面

## 最近重大改进

### 🛡 威胁检测增强
- 新增冲四、跳四、眠三等12种威胁模式识别
- 威胁权重优化：冲四(1200) > 活四(1000) > 活三(800)
- 双威胁检测：同时识别多个威胁组合

### ⚙ 算法优化
```python
# 动态资源分配示例
complexity = sum(1 for row in board for cell in row if cell != 0)
mcts_sims = min(400, 150 + complexity * 10)
pvs_depth = min(10, 5 + complexity // 5)
```

### 🔄 多进程协同
- 使用Manager共享内存提高进程间通信效率
- 精确超时控制：确保在时限内返回最佳结果
- 异常处理增强：单个进程失败不影响整体决策

### 🧩 评估函数升级
```python
# 新增连子奖励
for dr, dc in [(1,0),(0,1),(1,1),(1,-1)]:
    count = 1
    for i in range(1,5):
        if board[r+dr*i][c+dc*i] == player:
            count += 1
    score += count * 15  # 连子奖励
```

### 🧭 开局策略优化
- 扩展中心点候选集：从1个增加到17个
- 距离衰减影响图：近点权重大于远点
- 开局库支持：记录常见开局模式

## 安装与运行

### 系统要求
- Python 3.8+
- Pygame 2.0+

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/your-username/AI-Gomoku.git
cd AI-Gomoku

# 安装依赖
pip install pygame numpy

# 运行游戏
python AI_Gomoku.py
```

### 启动选项
```bash
# 启用Beta模式（包含专业对决和AI对AI）
python AI_Gomoku.py --beta

# 指定初始难度
python AI_Gomoku.py --difficulty hell
```

## 使用说明

1. 启动游戏后选择难度模式
2. 在棋盘上点击落子位置
3. 观察AI的实时分析和决策过程
4. 使用"Replay"按钮重新开始
5. 在Beta模式下体验"AI对AI"专业对决

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议，在保护代码权益的同时允许商业使用：

```text
Copyright 2023 Your Name

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

## 贡献与反馈

欢迎贡献代码和提出改进建议：

1. 提交Issue报告问题或建议
2. Fork仓库并提交Pull Request
3. 加入讨论：gomoku-ai@example.com

## 项目路线图

- [ ] GPU加速支持
- [ ] 神经网络模型集成
- [ ] 在线对战功能
- [ ] 开局库和学习系统
- [ ] 移动端适配

---

**体验智能五子棋的魅力，挑战你的策略极限！** [记住我们]([https://github.com/my-txz/AI-#](https://github.com/my-txz/AI-#)
