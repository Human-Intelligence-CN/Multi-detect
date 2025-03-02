import cv2
import sys
import torch
from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import QTimer

from main_window import Ui_MainWindow

# 定义一个函数，将OpenCV的图像格式转换为QImage格式
def convert2QImage(img):
    height, width, channel = img.shape
    bytes_per_line = width * channel
    return QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.model = torch.hub.load("./", "custom", path="runs/train/exp/weights/best.pt", source="local")
        self.timer = QTimer(self)  # 将定时器的parent设置为当前窗口
        self.timer.timeout.connect(self.video_pred)
        self.video = None
        self.is_playing = False
        self.class_names = self.load_class_names()  # 加载类别名称
        self.bind_slots()

    # 加载类别名称
    def load_class_names(self):
        # 假设类别名称存储在'coco.names'文件中
        with open('datasets/classes.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names

    # 定时器超时时的回调函数，用于处理视频帧
    def video_pred(self):
        if self.is_playing and self.video and self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                self.stop_video()
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.model(frame)
                image = results.render()[0]
                self.label.setPixmap(QPixmap.fromImage(convert2QImage(image)))
                self.update_text_edit(results)

    # 更新QTextEdit以显示标签和数量的函数
    def update_text_edit(self, results):
        labels_and_counts = {}
        for det in results.xyxy[0]:  # 假设results.xyxy[0]包含检测结果
            # det[-1]是类别ID，det[4]是置信度
            label_id = int(det[-1])
            if label_id < 0 or label_id >= len(self.class_names):
                continue  # 如果类别ID超出范围，则跳过
            label_name = self.class_names[label_id]
            confidence = det[4].item()
            if label_name not in labels_and_counts:
                labels_and_counts[label_name] = {'count': 1, 'confidence': confidence}
            else:
                labels_and_counts[label_name]['count'] += 1
                labels_and_counts[label_name]['confidence'] = (labels_and_counts[label_name]['confidence'] * (
                            labels_and_counts[label_name]['count'] - 1) + confidence) / labels_and_counts[label_name][
                                                                'count']

        text = ""
        for label_name, info in labels_and_counts.items():
            text += f"{label_name}: Count={info['count']}, Avg Confidence={info['confidence']:.2f}\n"
        self.textEdit.setText(text)

    # 打开或暂停视频的函数
    def open_video(self):
        if self.is_playing:
            self.stop_video(True)
            self.is_playing = False
        else:
            if not self.video or not self.video.isOpened():
                self.video = cv2.VideoCapture(0)
            if self.video and self.video.isOpened():
                self.timer.start(1)  # 调整定时器间隔为30毫秒
                self.is_playing = True
        self.update_button_text()

    def stop_video(self, clear_display=False):
        self.timer.stop()
        self.is_playing = False
        if self.video:
            self.video.release()
            self.video = None
        if clear_display:
            self.label.clear()

    def bind_slots(self):
        self.pushButton.clicked.connect(self.open_video)
        self.pushButton2 = QPushButton("保存文本", self)
        self.pushButton2.setGeometry(500, 410, 100, 30)  # 设置按钮位置和大小
        self.pushButton2.clicked.connect(self.save_text_to_file)
        self.pushButton3 = QPushButton("保存图片", self)
        self.pushButton3.setGeometry(500, 480, 100, 30)  # 设置按钮位置和大小
        self.pushButton3.clicked.connect(self.save_current_frame)

    # 更新按钮文本的函数，以反映当前是播放还是暂停状态
    def update_button_text(self):
        self.pushButton.setText("开始" if not self.is_playing else "暂停")

    # 保存文本到文件的函数
    def save_text_to_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "Text Files (*.txt)")
        if file_path:
            start_text = "START\n"
            end_text = "\nEND"
            content = self.textEdit.toPlainText()
            with open(file_path, 'w') as file:
                file.write(start_text + content + end_text)

    # 保存当前帧到文件的函数
    def save_current_frame(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_path:
            ret, frame = self.video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(file_path, frame)

# 主函数，创建QApplication对象，显示主窗口，并进入事件循环
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()