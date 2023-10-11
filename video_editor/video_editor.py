import sys, os
import typing
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from pathlib import Path
import cv2
import pdb
from time import sleep
import threading
import platform
import numpy as np

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
# video_editor_ui = uic.loadUiType("video_editor.ui")[0]
video_editor_ui = uic.loadUiType("video_editor_large.ui")[0]
# image_size = (640, 480)
image_size = (1280, 720)
video_save_ui = uic.loadUiType("video_save.ui")[0]

config_dir = Path(os.getenv('LOCALAPPDATA')) / 'pyqt' if platform.system() == "Windows" else Path.home() / '.pyqt'
config_file = str(config_dir.absolute() / 'video_editor.txt')
last_dir = ""

def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

def last_visited_dir() -> str:
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            last_dir = f.read()
            if os.path.exists(last_dir):
                return last_dir
    return str(Path.home())

def last_visited_dir_save(last_dir: str) -> None:
    if not config_dir.exists():
        config_dir.mkdir(parents=True)
    with open(config_file, 'w') as f:
        f.write(last_dir)
    return

class SavingWindowClass(QDialog, video_save_ui) :
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.geometry().width(), self.geometry().height())
        self.buttonBox.accepted.connect(self.save_video)
        self.buttonBox.rejected.connect(self.close)
        self.FindButton.clicked.connect(self.openDir)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.speedUp.valueChanged.connect(self.speedUp_valueChanged)
        self.FilenamLineEdit.textChanged.connect(lambda : self.pBar.setValue(0))
        self.saveSpeed = 1
        self.parent = parent
        self.fps = self.parent.video_fps[0]
        self.output_width, self.output_height = self.parent.video_sizes[0] # TODO
        self.blackframe = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)

    def speedUp_valueChanged(self):
        self.saveSpeed = self.speedUp.value()

    def openDir(self):
        # refer_dir = './' if self.parent.file_list[0] == '' else str(Path(self.parent.file_list[0]).parent.absolute())
        filename = QFileDialog.getSaveFileName(self, 'Save file', last_visited_dir())[0]
        if Path(filename).suffix != '.mp4':
            filename += '.mp4'
        self.FilenamLineEdit.setText(filename)

    def save_video(self):
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        filename = self.FilenamLineEdit.text()
        print(filename)
        output_video = cv2.VideoWriter(str(filename), fourcc, self.fps, (self.output_width, self.output_height))

        cutting_pair = []
        for i in range(len(self.parent.cutting_list) // 2):
            cutting_pair.append([self.parent.cutting_list[2*i], self.parent.cutting_list[2*i+1]])
        if len(self.parent.cutting_list) % 2 == 1:
            print(" ** Warning: cutting list has odd count -> ignore the last cutting point **")

        print(f"cutting_pair: {cutting_pair}")
        cutting_output_frame_num = 0
        real_output_frame_num = 0
        for s, e in cutting_pair:
            cutting_output_frame_num += e - s + 1
        real_output_frame_num = int(cutting_output_frame_num / self.saveSpeed)
        black_front_frame = self.BlackFront.value()
        black_rear_frame = self.BlackRear.value()
        total_output_frame_num = black_front_frame + real_output_frame_num + black_rear_frame
        print(f"total_output_frame_num: {total_output_frame_num} = B {black_front_frame} + {cutting_output_frame_num} / {self.saveSpeed} + B {black_rear_frame}")

        done_frame_num = 0
        for i in range(black_front_frame):
            # frame_out = cv2.cvtColor(self.blackframe, cv2.COLOR_RGB2BGR)
            output_video.write(self.blackframe)
            done_frame_num += 1
            self.pBar.setValue(int(done_frame_num/total_output_frame_num*100))
        for s, e in cutting_pair:
            for i in range(s, e+1):
                if done_frame_num % self.saveSpeed == 0:
                    frame_out = cv2.cvtColor(self.parent.frame_all[i], cv2.COLOR_RGB2BGR)
                    output_video.write(frame_out)
                done_frame_num += 1
                self.pBar.setValue(int(done_frame_num/total_output_frame_num*100))
        for i in range(black_rear_frame):
            # frame_out = cv2.cvtColor(self.blackframe, cv2.COLOR_RGB2BGR)
            output_video.write(self.blackframe)
            done_frame_num += 1
            self.pBar.setValue(int(done_frame_num/total_output_frame_num*100))
        self.pBar.setValue(100)
        output_video.release()
        print("Done !")

class WindowClass(QMainWindow, video_editor_ui) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.geometry().width(), self.geometry().height())
        self.FileName_Label.setText("* please load video first *")
        self.FrameNum_Label.setText("[         /]") # 9 spaces

        # 최종 확인 취소 버튼
        self.buttonBox.accepted.connect(self.SavingWindow)
        self.buttonBox.rejected.connect(self.close)

        # Video list 관리
        self.blockFrame1(False) # un-blocked
        self.listAddButton.clicked.connect(self.addList)
        self.listDelButton.clicked.connect(self.delList)
        self.listUpButton.clicked.connect(self.upList)
        self.listDownButton.clicked.connect(self.downList)
        self.listpBar.setVisible(False)
        self.LoadVideoCheckBox.stateChanged.connect(self.LoadVideo)

        # 재생 slider
        self.blockFrame2(True) # blocked
        self.playMode = False

        # self.VideoThread = QThread() # TODO - QThread?

        #self.slider.valueChanged.connect(lambda : self.showImage(self.slider.value()) or self.showFilename())
        self.slider.valueChanged.connect(self.slider_valueChanged)

        self.frameSpinBox.valueChanged.connect(self.frameSpinBox_valueChanged)
        self.speedUp.valueChanged.connect(self.speedUp_valueChanged)
        self.sliderLeftButton.clicked.connect(self.prevFrame)
        self.playButton.clicked.connect(self.playVideo)
        self.sliderRightButton.clicked.connect(self.nextFrame)

        # Cutter
        self.setCutterButton.clicked.connect(self.setCutter)
        self.delCutterButton.clicked.connect(self.delCutter)
        # self.fileLine.textChanged.connect(lambda : self.pBar.setValue(0))
        # self.fileLine.returnPressed.connect(self.cButtonFn)

        # 저장 변수
        self.file_list = []
        self.cutting_list = []
        self.frame_all = []
        self.video_sizes = []
        self.video_fps = []
        self.video_frame_num = []
        self.video_frame_accum = []
        self.total_frame_num = 0
        self.playspeed = 1

    def frameSpinBox_valueChanged(self):
        self.slider.setValue(self.frameSpinBox.value())

    def speedUp_valueChanged(self):
        self.playspeed = self.speedUp.value()

    def slider_valueChanged(self):
        self.frameSpinBox.setValue(self.slider.value())
        self.showImage(self.slider.value())

    def clear_video(self):
        self.frame_all = []
        self.video_sizes = []
        self.video_fps = []
        self.video_frame_num = []
        self.video_frame_accum = []
        self.total_frame_num = 0

        self.cutting_list = []

        self.ImageLabel.clear()
        self.slider.setValue(0) # run showImage
        self.slider.setMaximum(99)

        self.FileName_Label.setText("* please load video first *")
        self.FrameNum_Label.setText("[         /]") # 9 spaces



    def blockFrame1(self, blockBool):
        self.listWidget.blockSignals(blockBool)
        self.listAddButton.setDisabled(blockBool)
        self.listDelButton.setDisabled(blockBool)
        self.listUpButton.setDisabled(blockBool)
        self.listDownButton.setDisabled(blockBool)
        self.setAcceptDrops(not blockBool)

    def blockFrame2(self, blockBool):
        self.frameSpinBox.setDisabled(blockBool)
        self.speedUp.setDisabled(blockBool)
        self.slider.setDisabled(blockBool)
        self.sliderLeftButton.setDisabled(blockBool)
        self.sliderRightButton.setDisabled(blockBool)
        self.playButton.setDisabled(blockBool)
        self.setCutterButton.setDisabled(blockBool)
        self.delCutterButton.setDisabled(blockBool)

    def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()
    def dropEvent(self, event):
        dropfiles = [u.toLocalFile() for u in event.mimeData().urls()]
        for f in dropfiles:
            if f in self.file_list or not is_video(f):
                continue
            self.file_list.append(f)
            self.listWidget.addItem(f)
        if len(dropfiles) > 0 :
            last_visited_dir_save(str(Path(dropfiles[0]).parent))

    def addList(self):
        openfiles = QFileDialog.getOpenFileNames(self, 'Open file', last_visited_dir())[0]
        for f in openfiles:
            if f in self.file_list or not is_video(f):
                continue
            self.file_list.append(f)
            self.listWidget.addItem(f)
        if len(openfiles) > 0 :
            last_visited_dir_save(str(Path(openfiles[0]).parent))
    def delList(self):
        if len(self.listWidget.selectedItems()) > 0:
            selected_item = self.listWidget.selectedItems()[0]
            seleted_row = self.listWidget.row(selected_item)
            del self.file_list[seleted_row]
            self.listWidget.takeItem(seleted_row)
    def upList(self):
        selected_item = self.listWidget.selectedItems()[0]
        selected_row = self.listWidget.row(selected_item)
        if selected_row > 0:
            moved_item = self.listWidget.takeItem(selected_row - 1)
            self.listWidget.insertItem(selected_row - 1, selected_item)
            self.listWidget.insertItem(selected_row, moved_item)
            self.file_list[selected_row], self.file_list[selected_row - 1] = self.file_list[selected_row - 1], self.file_list[selected_row]
    def downList(self):
        selected_item = self.listWidget.selectedItems()[0]
        selected_row = self.listWidget.row(selected_item)
        if selected_row < self.listWidget.count() - 1:
            moved_item = self.listWidget.takeItem(selected_row + 1)
            self.listWidget.insertItem(selected_row + 1, selected_item)
            self.listWidget.insertItem(selected_row, moved_item)
            self.file_list[selected_row], self.file_list[selected_row + 1] = self.file_list[selected_row + 1], self.file_list[selected_row]

    def LoadVideo(self):
        if self.LoadVideoCheckBox.isChecked():
            self.blockFrame1(True) # block
            self.LoadVideoImage()
            self.blockFrame2(False) # un-block
            # TODO -  show 0 index image
        else :
            self.blockFrame2(True) # block
            self.blockFrame1(False) # un-block
            self.clear_video()

    def LoadVideoImage(self):
        for video_file in self.file_list:
            cap = cv2.VideoCapture(str(video_file))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            self.video_sizes.append((frame_width, frame_height))
            self.video_fps.append(fps)
            self.video_frame_num.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        # print(self.video_sizes)
        # print(self.video_fps)
        # print(self.video_frame_num)
        self.listpBar.setVisible(True)
        self.total_frame_num = sum(self.video_frame_num)
        self.listpBar.setMaximum(sum(self.video_frame_num))

        loading_cnt = 0
        for i, video_file in enumerate(self.file_list):
            cap = cv2.VideoCapture(str(video_file))
            est_frame_num = self.video_frame_num[i]
            real_frame_num = 0
            try_frame_num = 0
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    try_frame_num += 1
                    if ret:
                        # print(f"{done_idx}/{total_frame_num} in {video_file}")
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_all.append(frame)
                        real_frame_num += 1
                        loading_cnt += 1
                        self.listpBar.setValue(loading_cnt)
                    elif try_frame_num > est_frame_num:
                        break
                    else:
                        pass
            if est_frame_num != real_frame_num:
                print(f"Warning: {video_file} has invalid frame: est {est_frame_num} -> real {real_frame_num}")
                self.video_frame_num[i] = real_frame_num
            cap.release()
        self.total_frame_num = sum(self.video_frame_num)
        self.slider.setMaximum(self.total_frame_num - 1)
        self.video_frame_accum = [ sum(self.video_frame_num[:x+1]) for x in range(len(self.video_frame_num)) ]
        assert(loading_cnt == self.total_frame_num)
        self.listpBar.setVisible(False)
        self.FrameNum_Label.setText(f"[         /{self.total_frame_num-1}]") # 9 spaces
        self.showImage(0)

    def playVideo(self):
        self.playMode = not self.playMode
        if self.playMode:
            self.VideoThread = threading.Thread(target=self.playVideoThread, daemon=True)
            self.VideoThread.start() # run playVideoThread
    def prevFrame(self):
        self.slider.setValue(max(self.slider.value() - self.playspeed, 0))
    def nextFrame(self):
        self.slider.setValue(min(self.slider.value() + self.playspeed, len(self.frame_all) - 1))

    def playVideoThread(self):
        while self.playMode:
            current_frame = self.slider.value()
            if current_frame == len(self.frame_all) - 1:
                self.playMode = False
                break
            self.slider.setValue(min(current_frame + self.playspeed, len(self.frame_all) - 1))
            sleep(1 / self.video_fps[0])

    def showImage(self, frame_idx):
        if frame_idx >= len(self.frame_all):
            return
        #print(f"[{frame_idx:04d}/{self.total_frame_num-1}]")
        self.showFilename()
        img = self.frame_all[frame_idx]
        qImg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        p = pixmap.scaled(image_size[0], image_size[1], Qt.KeepAspectRatio)
        self.ImageLabel.setPixmap(p)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prevFrame()
        elif event.key() == Qt.Key_Right:
            self.nextFrame()
        elif event.key() == Qt.Key_Space:
            self.playVideo()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 125, 0))
        painter.setBrush(QColor(255, 255, 0))

        player_geo = self.PlayFrame.geometry()
        slider_geo = self.slider.geometry()
        sx, sy, sw, sh = player_geo.x() + slider_geo.x(), player_geo.y() + slider_geo.y(), slider_geo.width(), slider_geo.height()
        # sx, sy, sw, sh = slider_geo.x(), slider_geo.y(), slider_geo.width(), slider_geo.height()
        frame_max = self.slider.maximum()
        left_padding = 5
        right_padding = 4
        sw = sw - left_padding - right_padding

        for frame in self.cutting_list:
            x1 = left_padding + sx + int(sw * frame / frame_max)
            y1 = sy + 2
            y2 = sy + sh - 4
            painter.drawLine(QPoint(x1, y1), QPoint(x1, y2))

        cutting_pair = []
        for i in range(len(self.cutting_list) // 2):
            cutting_pair.append([self.cutting_list[2*i], self.cutting_list[2*i+1]])
        # if len(self.cutting_list) % 2 == 1:
        #     cutting_pair.append([self.cutting_list[-1], self.cutting_list[-1] + 1])

        for start_frame, end_frame in cutting_pair:
            x1 = left_padding + sx + int(sw * start_frame / frame_max)
            y1 = sy + 2
            bw = int(sw * (end_frame - start_frame) / frame_max)
            bh = 8
            painter.drawRect(QRect(x1, y1, bw, bh))

    def showFilename(self): # TODO - bug in VideoThread
        curr_idx = self.slider.value()
        file_index = [ curr_idx > fnum for fnum in self.video_frame_accum ].index(False)
        # if curr_idx = 10, video_frame_accum = [100, 200, 300] -> [False, False, False] -> file_index = 0
        # if curr_idx = 110, video_frame_accum = [100, 200, 300] -> [True, False, False] -> file_index = 1
        filename = Path(self.file_list[file_index]).name
        file_frame_idx = curr_idx - self.video_frame_accum[file_index-1] if file_index > 0 else curr_idx
        self.FileName_Label.setText(f"{filename}:{file_frame_idx}")

    def setCutter(self):
        cutting_point = self.slider.value()
        if cutting_point not in self.cutting_list:
            self.cutting_list.append(cutting_point)
        self.cutting_list.sort()
        print(self.cutting_list)
        self.update()

    def delCutter(self):
        curr_point = self.slider.value()
        del_cutting_point_candi = -1
        del_cutting_point_idx = 0
        diff = self.total_frame_num
        for i, cutting_point in enumerate(self.cutting_list):
            if abs(curr_point - cutting_point) < diff:
                diff = abs(curr_point - cutting_point)
                del_cutting_point_candi = cutting_point
                del_cutting_point_idx = i

        if del_cutting_point_candi != -1: # and diff < 20:
            del_cutting_point = del_cutting_point_candi
            del self.cutting_list[del_cutting_point_idx]
            self.slider.setValue(del_cutting_point)
        print(self.cutting_list)
        self.update()

    def SavingWindow(self):
        self.SavingWindow = SavingWindowClass(self)
        self.SavingWindow.exec_()


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_()

