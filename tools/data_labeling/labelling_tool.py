import os
import sys

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QFileDialog, QSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pynput import keyboard
from pynput.keyboard import Key

"""
IMULabelTool - 新版 IMU 数据标注工具
=====================================

概述
----
这个工具用于手动标注 IMU（惯性测量单元）数据中的敲击事件。
支持新的 IMU 数据格式，包括加速度计、角加速度和陀螺仪数据。
使用固定步长导航替代基于已有标签的跳转方式。

主要功能
--------
1. **加载 CSV 文件**：
   - 支持加载包含 IMU 数据的 CSV 文件
   - 自动检测并适配数据格式：ax, ay, az, agx, agy, agz, gx, gy, gz, roll, pitch, yaw
   - 如果没有 Label 列，自动创建并初始化为 0

2. **数据可视化**：
   - 使用两个子图分别显示加速度计数据 (ax, ay, az) 和陀螺仪数据 (gx, gy, gz)
   - 显示当前位置周围的窗口数据（默认 90 个数据点）

3. **固定步长导航**：
   - 使用"上一个"和"下一个"按钮进行固定步长移动（默认 20 点）
   - 支持用户自定义步长大小

4. **手动标注**：
   - 用户可以通过输入框手动设置起止行号，并点击"标记"按钮标注敲击事件
   - 支持通过点击图表快速设置起止行号

5. **文件操作**：
   - 默认打开 /Users/aksea/Project/Python/PhoneClick/data/raw 目录
   - 保存时自动添加 _labeled 后缀并保存到 data/labeled 目录

使用流程
--------
1. 点击"加载 CSV 文件"加载数据
2. 使用"上一个"/"下一个"按钮浏览数据
3. 通过输入框或点击图表设置标注范围
4. 点击"标记"应用标注
5. 点击"保存标注"保存结果

键盘快捷键
-----------
- space：切换点击模式
  - 自定义间隔模式：设置起始点 → 设置终止点 → 执行标记
  - 默认间隔模式：设置起始点 → 执行标记（自动设置终止点为起始点+15）
- left：向前移动一个步长
- right：向后移动一个步长

间隔模式
--------
- 自定义间隔模式：需要手动设置起始点和终止点
- 默认间隔模式：设置起始点后自动设置终止点为起始点+15
"""


class IMULabelTool(QWidget):
    def __init__(self):
        super().__init__()
        self.click_marker = None
        self.setWindowTitle("IMU 数据标注工具 - 新版")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 初始化状态
        self.df = None  # 加载的 DataFrame
        self.opened_file = None  # 记录打开的文件路径
        self.current_position = 0  # 当前查看位置（行号）
        self.window_size = 90  # 显示窗口的大小
        self.step_size = 20  # 导航步长
        
        # 预期的数据列名
        self.expected_columns = [
            'ax', 'ay', 'az',  # 加速度计
            'agx', 'agy', 'agz',  # 角加速度（暂不显示）
            'gx', 'gy', 'gz',  # 陀螺仪
            'roll', 'pitch', 'yaw'  # 姿态（暂不显示）
        ]

        # 可视化画布设置，两个子图
        self.canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.ax1 = self.canvas.figure.add_subplot(2, 1, 1)  # 加速度图
        self.ax2 = self.canvas.figure.add_subplot(2, 1, 2)  # 陀螺仪图
        
        # 调整子图间距避免重合
        self.canvas.figure.subplots_adjust(hspace=0.3)

        # 按钮和输入框
        self.load_button = QPushButton("加载 CSV 文件")
        self.prev_button = QPushButton("← 上一个")  # 向前移动
        self.next_button = QPushButton("下一个 →")  # 向后移动
        self.mark_button = QPushButton("标记")  # 标记选定范围
        self.clear_button = QPushButton("清除标记")  # 清除当前窗口标记
        self.save_button = QPushButton("保存标注")  # 保存修改后的 CSV
        self.interval_mode_button = QPushButton("默认间隔模式")  # 切换间隔模式

        self.start_input = QLineEdit()  # 输入起始行号
        self.end_input = QLineEdit()  # 输入结束行号
        
        # 步长设置
        self.step_input = QSpinBox()
        self.step_input.setMinimum(1)
        self.step_input.setMaximum(1000)
        self.step_input.setValue(self.step_size)
        self.step_button = QPushButton("设置步长")

        self.range_label = QLabel("标记范围（行号）:")
        self.step_label = QLabel("导航步长:")
        self.info_label = QLabel("状态：未加载文件")
        self.position_label = QLabel("位置：-")

        self.click_mode = 0  # 图中点击操作的状态：0 = 不做任何事，1 = 设置 start，2 = 设置 end
        self.default_interval_mode = False  # 默认间隔模式：False = 自定义间隔，True = 默认间隔(15)

        # 初始化 UI 布局和事件绑定
        self.init_ui()
        self.bind_events()

        # 图表中鼠标点击事件
        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

    def init_ui(self):
        """构建界面布局"""
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)

        # 导航按钮行
        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.load_button)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        nav_layout.addWidget(self.step_label)
        nav_layout.addWidget(self.step_input)
        nav_layout.addWidget(self.step_button)
        layout.addLayout(nav_layout)

        # 标记行输入 + 按钮行
        mark_layout = QHBoxLayout()
        mark_layout.addWidget(self.range_label)
        mark_layout.addWidget(self.start_input)
        mark_layout.addWidget(self.end_input)
        mark_layout.addWidget(self.mark_button)
        mark_layout.addWidget(self.clear_button)
        mark_layout.addWidget(self.interval_mode_button)
        layout.addLayout(mark_layout)

        # 状态和保存行
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.save_button)
        status_layout.addWidget(self.position_label)
        status_layout.addWidget(self.info_label)
        layout.addLayout(status_layout)

        self.setLayout(layout)

    def bind_events(self):
        """绑定按钮事件"""
        self.load_button.clicked.connect(self.load_csv)
        self.next_button.clicked.connect(self.move_forward)
        self.prev_button.clicked.connect(self.move_backward)
        self.mark_button.clicked.connect(self.mark_data)
        self.clear_button.clicked.connect(self.clear_mark)
        self.save_button.clicked.connect(self.save_csv)
        self.step_button.clicked.connect(self.set_step_size)
        self.interval_mode_button.clicked.connect(self.toggle_interval_mode)

    def set_step_size(self):
        """设置步长大小"""
        self.step_size = self.step_input.value()
        self.info_label.setText(f"步长已设置为 {self.step_size}")

    def toggle_interval_mode(self):
        """切换间隔模式"""
        self.default_interval_mode = not self.default_interval_mode
        if self.default_interval_mode:
            self.interval_mode_button.setText("自定义间隔模式")
            self.info_label.setText("已切换到默认间隔模式：设置起始点后自动设置终止点为起始点+15")
        else:
            self.interval_mode_button.setText("默认间隔模式")
            self.info_label.setText("已切换到自定义间隔模式：需要手动设置起始点和终止点")
        # 重置点击模式
        self.click_mode = 0

    def load_csv(self):
        """加载 CSV 文件并初始化"""
        # 设置默认打开路径
        default_path = "/Users/aksea/Project/Python/PhoneClick/data/raw"
        if not os.path.exists(default_path):
            default_path = ""
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", default_path, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                self.opened_file = file_path
                self.df = pd.read_csv(file_path)
                
                # 检查数据格式
                self.check_and_adapt_data_format()
                
                # 初始化位置
                self.current_position = self.window_size // 2
                
                # 显示数据
                self.show_window_around(self.current_position)
                
                total_rows = len(self.df)
                self.info_label.setText(f"文件加载成功，共 {total_rows} 行数据")
                
            except Exception as e:
                self.info_label.setText(f"加载失败：{str(e)}")

    def check_and_adapt_data_format(self):
        """检查并适配数据格式"""
        # 检查必要的数据列
        missing_cols = []
        for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
            if col not in self.df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"缺少必要的数据列: {', '.join(missing_cols)}")
        
        # 如果没有 Label 列，创建一个
        if 'Label' not in self.df.columns:
            self.df['Label'] = 0
            self.info_label.setText("已创建 Label 列")

    def show_window_around(self, center):
        """显示指定中心位置周围的数据窗口"""
        try:
            if self.df is None:
                return
                
            # 计算窗口范围
            half = self.window_size // 2
            start = max(0, center - half)
            end = min(len(self.df), start + self.window_size)
            start = end - self.window_size if end - start < self.window_size else start

            window = self.df.iloc[start:end]

            # 清除之前的图像
            self.ax1.clear()
            self.ax2.clear()
            self.click_marker = None
            self.click_mode = 0

            # 绘制加速度计数据
            self.ax1.plot(window.index, window['ax'], label='ax', color='blue')
            self.ax1.plot(window.index, window['ay'], label='ay', color='green')
            self.ax1.plot(window.index, window['az'], label='az', color='cyan')
            self.ax1.plot(window.index, window['Label'] * 10, label='Label', linestyle='--', color='red')
            self.ax1.set_title('加速度计数据 (Accelerometer)')
            self.ax1.legend(loc='upper left')
            self.ax1.grid()

            # 绘制陀螺仪数据
            self.ax2.plot(window.index, window['gx'], label='gx', color='orange')
            self.ax2.plot(window.index, window['gy'], label='gy', color='red')
            self.ax2.plot(window.index, window['gz'], label='gz', color='magenta')
            self.ax2.plot(window.index, window['Label'] * 10, label='Label', linestyle='--', color='red')
            self.ax2.set_title('陀螺仪数据 (Gyroscope)')
            self.ax2.legend(loc='upper left')
            self.ax2.grid()

            self.canvas.draw()

            # 更新位置信息
            total_rows = len(self.df)
            self.position_label.setText(f"位置：{center}/{total_rows} (窗口：{start}-{end})")

            # 自动识别窗口内 Label=1 的起止行
            label_window = self.df.iloc[start:end]['Label']
            ones_indices = label_window[label_window == 1].index.tolist()
            if ones_indices:
                self.start_input.setText(str(ones_indices[0]))
                self.end_input.setText(str(ones_indices[-1]))
            else:
                self.start_input.setText("")
                self.end_input.setText("")
                
        except Exception as e:
            self.info_label.setText(f"显示错误：{str(e)}")

    def move_forward(self):
        """向后移动一个步长"""
        if self.df is None:
            return
            
        max_position = len(self.df) - self.window_size // 2
        self.current_position = min(max_position, self.current_position + self.step_size)
        self.show_window_around(self.current_position)

    def move_backward(self):
        """向前移动一个步长"""
        if self.df is None:
            return
            
        min_position = self.window_size // 2
        self.current_position = max(min_position, self.current_position - self.step_size)
        self.show_window_around(self.current_position)

    def mark_data(self):
        """将输入框中的 start-end 行标记为 Label=1"""
        if self.df is None:
            return
            
        try:
            start = int(self.start_input.text())
            end = int(self.end_input.text())
            
            if start > end:
                start, end = end, start
                
            # 清除当前窗口的标记
            half = self.window_size // 2
            window_start = max(0, self.current_position - half)
            window_end = min(len(self.df), window_start + self.window_size)
            self.df.loc[window_start:window_end-1, 'Label'] = 0
            
            # 设置新的标记
            self.df.loc[start:end, 'Label'] = 1
            self.info_label.setText(f"标记成功：行 {start} 到 {end}")
            self.show_window_around(self.current_position)
            
        except ValueError:
            self.info_label.setText("标记失败：请输入有效的行号")
        except Exception as e:
            self.info_label.setText(f"标记失败：{str(e)}")

    def clear_mark(self):
        """清除当前窗口中的所有标记"""
        if self.df is None:
            return
            
        half = self.window_size // 2
        window_start = max(0, self.current_position - half)
        window_end = min(len(self.df), window_start + self.window_size)
        self.df.loc[window_start:window_end-1, 'Label'] = 0
        self.info_label.setText("当前窗口的标记已清除")
        self.show_window_around(self.current_position)

    def save_csv(self):
        """保存标注后的数据"""
        if self.df is None:
            return
            
        if self.opened_file:
            # 构建默认保存路径
            base_name = os.path.basename(self.opened_file)
            name_without_ext = os.path.splitext(base_name)[0]
            
            # 添加 _labeled 后缀
            if not name_without_ext.endswith('_labeled'):
                default_file_name = f"{name_without_ext}_labeled.csv"
            else:
                default_file_name = f"{name_without_ext}.csv"
            
            # 设置保存目录
            project_root = "/Users/aksea/Project/Python/PhoneClick"
            save_dir = os.path.join(project_root, "data", "labeled")
            
            # 创建目录如果不存在
            os.makedirs(save_dir, exist_ok=True)
            
            default_path = os.path.join(save_dir, default_file_name)
        else:
            default_path = ""

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存 CSV 文件", default_path, "CSV Files (*.csv)"
        )
        
        if file_path:
            try:
                # 确保保存包含 Label 列
                self.df.to_csv(file_path, index=False)
                self.info_label.setText("保存成功")
            except Exception as e:
                self.info_label.setText(f"保存失败：{str(e)}")

    def on_plot_click(self, event):
        """图表点击事件处理"""
        if event.inaxes:
            x = int(round(event.xdata))
            y = event.ydata
            self.info_label.setText(f"点击坐标: x = {x}, y = {y:.2f}")

            # 移除之前的标记
            if self.click_marker is not None:
                self.click_marker.remove()

            # 添加新的标记
            self.click_marker = event.inaxes.plot(x, y, 'ro', markersize=8)[0]
            self.canvas.draw()

            # 根据点击模式设置输入框
            if self.click_mode == 1:
                self.start_input.setText(str(x))
                if self.default_interval_mode:
                    # 默认间隔模式：自动设置终止点为起始点+15
                    end_value = x + 15
                    self.end_input.setText(str(end_value))
            elif self.click_mode == 2:
                self.end_input.setText(str(x))

    def on_press(self, key):
        """键盘快捷键处理"""
        try:
            if key == Key.space:
                if self.default_interval_mode:
                    # 默认间隔模式：设置起始点 → 执行标记
                    if self.click_mode == 0:
                        self.click_mode = 1
                        self.info_label.setText("点击图表：设置起始行号")
                    else:
                        self.mark_data()
                        self.click_mode = 0
                        self.info_label.setText("标记已执行")
                else:
                    # 自定义间隔模式：设置起始点 → 设置终止点 → 执行标记
                    self.click_mode = (self.click_mode + 1) % 3
                    if self.click_mode == 1:
                        self.info_label.setText("点击图表：设置起始行号")
                    elif self.click_mode == 2:
                        self.info_label.setText("点击图表：设置结束行号")
                    else:
                        self.mark_data()
                        self.info_label.setText("标记已执行")
            elif key == Key.left:
                self.move_backward()
            elif key == Key.right:
                self.move_forward()
        except AttributeError:
            pass

    def on_release(self, key):
        """按键释放处理"""
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = IMULabelTool()

    # 设置键盘监听
    listener = keyboard.Listener(
        on_press=window.on_press,
        on_release=window.on_release
    )
    listener.start()

    window.show()
    sys.exit(app.exec_())