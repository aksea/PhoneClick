import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QFileDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from pynput import keyboard
from pynput.keyboard import Key

"""
IMULabelTool - IMU 数据标注工具
===============================

概述
----
这个工具用于辅助标注 IMU（惯性测量单元）数据中的某些动作片段，特别是手指敲击等事件。
它从 CSV 文件中读取 IMU 数据（包括加速度计和陀螺仪数据），并识别 `Label` 列中从 0 到 1 的跳变点，作为动作的起始点。
用户可以浏览这些标注点周围的窗口数据，并手动修改标注的起止范围。

主要功能
--------
1. **加载 CSV 文件**：
   - 支持加载包含 `Timestamp`, `AX`, `AY`, `AZ`, `GX`, `GY`, `GZ`, `Label` 列的 CSV 文件。
   - 自动检测 `Label` 从 0 到 1 的跳变点，作为初始标注位置。

2. **数据可视化**：
   - 使用 Matplotlib 显示加速度和角速度数据，分为两个子图。
   - 显示当前标注点周围的窗口数据（默认 90 行）。

3. **导航与浏览**：
   - 使用“上一个”和“下一个”按钮在不同的标注点之间切换。
   - 自动显示当前窗口内 `Label=1` 的起止行号。

4. **手动标注**：
   - 用户可以通过输入框手动设置起止行号，并点击“标记”按钮将该范围的 `Label` 设为 1。
   - 支持通过点击图表快速设置起止行号（需先按空格键切换模式）。

5. **清除标记**：
   - 点击“清除标记”按钮将当前窗口内的所有 `Label` 设为 0。

6. **保存标注**：
   - 将修改后的数据保存为新的 CSV 文件，文件名默认为原文件名后缀 `_relabelling`。

7. **键盘快捷键**：
   - `space`：切换点击模式（设置起始点 → 设置终止点 → 自动标记）。
   - `left`：切换到上一个标注点。
   - `right`：切换到下一个标注点。

使用流程
--------
1. **加载文件**：
   - 点击“加载 CSV 文件”按钮，选择包含 IMU 数据的 CSV 文件。

2. **浏览标注点**：
   - 使用“上一个”和“下一个”按钮切换到不同的标注点。
   - 观察当前窗口的数据可视化，查看自动识别的 `Label=1` 范围。

3. **修改标注**：
   - **手动输入**：在输入框中输入新的起止行号。
   - **点击设置**：按 `space` 键切换到“设置起始点”或“设置终止点”模式，然后点击图表选择行号。
   - 点击“标记”按钮应用修改。

4. **清除标记**：
   - 点击“清除标记”按钮，将当前窗口内的所有 `Label` 设为 0。

5. **保存数据**：
   - 点击“保存标注”按钮，选择保存路径，生成新的 CSV 文件。

界面说明
--------
- **加载 CSV 文件**：选择并加载数据文件。
- **← 上一个 / 下一个 →**：在标注点之间切换。
- **标记**：根据输入框中的起止行号设置 `Label=1`。
- **清除标记**：将当前窗口的 `Label` 全部设为 0。
- **保存标注**：保存修改后的数据。
- **标记范围（行号）**：显示和输入起止行号。
- **状态信息**：显示当前操作状态和提示。

键盘快捷键
-----------
- **`space`**：切换点击模式：
  - 第一次按下：进入“设置起始点”模式，点击图表将设置起始行号。
  - 第二次按下：进入“设置终止点”模式，点击图表将设置终止行号。
  - 第三次按下：执行“标记”操作，并返回“无操作”模式。
- **`left`**：切换到上一个标注点。
- **`right`**：切换到下一个标注点。

注意事项
--------
- CSV 文件必须包含 `Label` 列，且 `Label` 值为 0 或 1。
- 窗口大小（`window_size`）默认为 90 行，可在代码中调整。
- 点击图表时，会在点击位置显示红点标记，并更新状态栏信息。
- 保存文件时，建议使用默认文件名以保持一致性。

示例操作
--------
1. 加载 CSV 文件。
2. 点击“下一个”查看第一个标注点。
3. 观察图表，找到动作的实际起止点。
4. 按 `space` 进入“设置起始点”模式，点击图表选择起始行。
5. 再次按 `space` 进入“设置终止点”模式，点击图表选择终止行。
6. 再次按 `space` 执行“标记”操作。
7. 点击“保存标注”保存修改后的数据。

"""


class IMULabelTool(QWidget):
    def __init__(self):
        super().__init__()
        self.click_marker = None
        self.setWindowTitle("IMU 数据标注工具")

        # 初始化状态
        self.df = None  # 加载的 DataFrame
        self.opened_file = None  # 记录打开的文件路径
        self.label_indices = []  # 所有 label=1 起始位置的索引
        self.current_index = 0  # 当前正在查看的标注点在 label_indices 中的索引
        self.window_size = 90  # 显示窗口的大小（显示当前点的上下数据）

        # 可视化画布设置，两个子图
        self.canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.ax1 = self.canvas.figure.add_subplot(2, 1, 1)  # 加速度图
        self.ax2 = self.canvas.figure.add_subplot(2, 1, 2)  # 角速度图

        # 按钮和输入框
        self.load_button = QPushButton("加载 CSV 文件")
        self.prev_button = QPushButton("← 上一个")  # 切换到上一个标注点
        self.next_button = QPushButton("下一个 →")  # 切换到下一个标注点
        self.mark_button = QPushButton("标记")  # 标记当前窗口中的起始和结束行为 Label=1
        self.clear_button = QPushButton("清除标记")  # 新增：清除当前窗口的标记
        self.save_button = QPushButton("保存标注")  # 保存修改后的 CSV

        self.start_input = QLineEdit()  # 输入起始行号
        self.end_input = QLineEdit()  # 输入结束行号

        self.range_label = QLabel("标记范围（行号）:")
        self.info_label = QLabel("状态：未加载文件")

        self.click_mode = 0  # 图中点击操作的状态：0 = 不做任何事，1 = 设置 start，2 = 设置 end

        # 初始化 UI 布局和事件绑定
        self.init_ui()
        self.bind_events()

        # 图表中鼠标点击事件：用于选择起始/结束行号
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def init_ui(self):
        # 构建界面布局
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # 导航按钮行
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.load_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        # 标记行输入 + 按钮行
        mark_layout = QHBoxLayout()
        mark_layout.addWidget(self.range_label)
        mark_layout.addWidget(self.start_input)
        mark_layout.addWidget(self.end_input)
        mark_layout.addWidget(self.mark_button)
        mark_layout.addWidget(self.clear_button)  # 新增：清除标记按钮
        layout.addLayout(mark_layout)

        # 保存行
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.save_button)
        save_layout.addWidget(self.info_label)
        layout.addLayout(save_layout)

        self.setLayout(layout)

    def bind_events(self):
        # 将按钮点击事件绑定到各自函数
        self.load_button.clicked.connect(self.load_csv)
        self.next_button.clicked.connect(self.next_label)
        self.prev_button.clicked.connect(self.prev_label)
        self.mark_button.clicked.connect(self.mark_data)
        self.clear_button.clicked.connect(self.clear_mark)  # 新增：绑定清除标记事件
        self.save_button.clicked.connect(self.save_csv)

    def load_csv(self):
        # 加载 CSV 文件并初始化标签起始点
        file_path, _ = QFileDialog.getOpenFileName(self, "选择 CSV 文件", "", "CSV Files (*.csv)")
        if file_path:
            self.opened_file = file_path  # 记录打开的文件路径
            self.df = pd.read_csv(file_path)
            if 'Label' not in self.df.columns:
                self.info_label.setText("错误：CSV 缺少 Label 列")
                return

            # 找出所有 label 从 0 -> 1 的起始点（边缘检测）
            label_values = self.df['Label'].values
            d_label = np.diff(label_values, prepend=0)  # prepend=0 to handle the first element
            self.label_indices = np.where(d_label == 1)[0]
            self.label_indices = self.label_indices.tolist()

            self.current_index = 0
            if self.label_indices:
                self.show_window_around(self.label_indices[self.current_index])
                self.info_label.setText(f"文件加载成功，找到 {len(self.label_indices)} 个标注起始点")
            else:
                self.info_label.setText("文件加载成功，但没有标注起始点")

    def show_window_around(self, center):
        try:
            # 根据当前 label 中心点，显示前后一定窗口范围的数据
            if self.df is None:
                return
            half = self.window_size // 2
            start = max(0, center - half)
            end = min(len(self.df), start + self.window_size)
            start = end - self.window_size if end - start < self.window_size else start

            window = self.df.iloc[start:end]

            self.ax1.clear()
            self.ax2.clear()
            self.click_marker = None
            self.click_mode = 0

            # 绘制加速度图
            self.ax1.plot(window.index, window['AX'], label='AX', color='blue')
            self.ax1.plot(window.index, window['AY'], label='AY', color='green')
            self.ax1.plot(window.index, window['AZ'], label='AZ', color='cyan')
            self.ax1.plot(window.index, window['Label'] * 50, label='Label', linestyle='--')
            self.ax1.set_title('Accelerometer')
            self.ax1.legend(loc='upper left')
            self.ax1.grid()

            # 绘制角速度图
            self.ax2.plot(window.index, window['GX'], label='GX', color='orange')
            self.ax2.plot(window.index, window['GY'], label='GY', color='red')
            self.ax2.plot(window.index, window['GZ'], label='GZ', color='magenta')
            self.ax2.plot(window.index, window['Label'] * 50, label='Label', linestyle='--')
            self.ax2.set_title('Gyroscope')
            self.ax2.legend(loc='upper left')
            self.ax2.grid()

            self.canvas.draw()

            # 更新窗口显示信息
            total = len(self.label_indices)
            current = self.current_index + 1
            self.info_label.setText(f"当前是第 {current} 个标注点，共 {total} 个")

            # 自动识别窗口内 Label=1 的起止行，填入输入框
            label_window = self.df.iloc[start:end]['Label']
            ones_indices = label_window[label_window == 1].index.tolist()
            if ones_indices:
                relative_start = ones_indices[0]
                relative_end = ones_indices[-1]
                self.start_input.setText(str(relative_start))
                self.end_input.setText(str(relative_end))
            else:
                self.start_input.setText("")
                self.end_input.setText("")
        except  Exception as e:
            raise e  # 抛出异常，显示详细错误信息

    def next_label(self):
        # 跳转到下一个标注点
        if self.current_index + 1 < len(self.label_indices):
            self.current_index += 1
            self.show_window_around(self.label_indices[self.current_index])
        else:
            self.info_label.setText("已经到最后一个标注点了")

    def prev_label(self):
        # 跳转到上一个标注点
        if self.df is None or not self.label_indices:
            return
        self.current_index = max(self.current_index - 1, 0)
        self.show_window_around(self.label_indices[self.current_index])

    def mark_data(self):
        # 将输入框中的 start-end 行标记为 Label=1，其余置为 0
        if self.df is None:
            return
        try:
            start = int(self.start_input.text())
            end = int(self.end_input.text())
            if start > end:
                start, end = end, start
            center = self.label_indices[self.current_index]
            half = self.window_size // 2
            window_start = max(0, center - half)
            window_end = min(len(self.df), window_start + self.window_size)
            self.df.loc[window_start:window_end, 'Label'] = 0
            self.df.loc[start:end, 'Label'] = 1
            self.info_label.setText(f"标记成功：行 {start} 到 {end}")
            self.show_window_around(center)
        except:
            self.info_label.setText("标记失败：请输入正确的起始和结束位置")

    def clear_mark(self):
        # 清除当前窗口中的所有标记（将 Label 设为 0）
        if self.df is None:
            return
        center = self.label_indices[self.current_index]
        half = self.window_size // 2
        window_start = max(0, center - half)
        window_end = min(len(self.df), window_start + self.window_size)
        self.df.loc[window_start:window_end, 'Label'] = 0
        self.info_label.setText("当前窗口的标记已清除")
        self.show_window_around(center)

    def save_csv(self):
        if self.df is not None:
            if self.opened_file:
                # 获取文件名（不含路径）
                base_name = os.path.basename(self.opened_file)
                # 去掉扩展名
                name_without_ext = os.path.splitext(base_name)[0]
                # 尝试提取 "数字_描述" 部分
                parts = name_without_ext.split('_')
                if parts[-1] == 'relabelling':
                    default_file_name = f"{name_without_ext}.csv"
                else:
                    # 如果不符合格式，直接追加 "_relabelling"
                    default_file_name = f"{name_without_ext}_relabelling.csv"
                # 获取原文件目录
                default_dir = os.path.dirname(self.opened_file)
                # 组合默认路径
                default_path = os.path.join(default_dir, default_file_name)
            else:
                default_path = ""  # 未打开文件时，默认路径为空
            file_path, _ = QFileDialog.getSaveFileName(self, "保存 CSV 文件", default_path, "CSV Files (*.csv)")
            if file_path:
                try:
                    self.df.to_csv(file_path, index=False)
                    self.info_label.setText("保存成功")
                except Exception as e:
                    self.info_label.setText(f"保存失败：{str(e)}")

    def on_plot_click(self, event):
        # 图表中鼠标点击事件：获取点击的 x 位置（行号）
        if event.inaxes:
            x = int(round(event.xdata))
            y = event.ydata
            self.info_label.setText(f"点击坐标: x = {x}, y = {y:.2f}")

            # 可视化红点标记
            if self.click_marker is not None and hasattr(self, 'click_marker'):
                self.click_marker.remove()

            self.click_marker = event.inaxes.plot(x, y, 'ro')[0]
            self.canvas.draw()

            # 如果处于设置起始/结束状态，则写入输入框
            if self.click_mode == 1:
                self.start_input.setText(str(x))
            elif self.click_mode == 2:
                self.end_input.setText(str(x))

    def on_press(self, key):
        # 空格键切换点击状态：设置起点 -> 终点 -> 自动标记
        # 支持键盘左右箭头切换上下标注点
        try:
            if key == Key.space:
                self.click_mode += 1
                self.click_mode %= 3
                if self.click_mode == 1:
                    self.info_label.setText("点击图表：将设置起始行号")
                elif self.click_mode == 2:
                    self.info_label.setText("点击图表：将设置结束行号")
                else:
                    self.mark_data()
                    self.info_label.setText("点击图表：不进行标记设置,对内容进行标记修正")
            elif key == Key.left:
                self.prev_label()
                print("点击上一标注点")
            elif key == Key.right:
                self.next_label()
                print("点击下一标注点")
        except AttributeError:
            pass

    def on_release(self, key):
        pass  # 可拓展：释放按键时做处理


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IMULabelTool()

    listener = keyboard.Listener(
        on_press=window.on_press,
        on_release=window.on_release
    )
    listener.start()

    window.show()
    sys.exit(app.exec_())
