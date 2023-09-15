import sys
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

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
video_editor_ui = uic.loadUiType("video_editor.ui")[0]
video_save_ui = uic.loadUiType("video_save.ui")[0]

def is_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']

class SavingWindowClass(QDialog, video_save_ui) :
    def __init__(self, parent):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(self.geometry().width(), self.geometry().height())
        self.buttonBox.accepted.connect(self.save_video)
        self.buttonBox.rejected.connect(self.close)
        self.FindButton.clicked.connect(self.openDir)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.parent = parent

    def openDir(self):
        refer_dir = './' if self.parent.file_list[0] == '' else str(Path(self.parent.file_list[0]).parent.absolute())
        self.filename = QFileDialog.getSaveFileName(self, 'Save file', refer_dir)[0]
        if Path(self.filename).suffix != '.mp4':
            self.filename += '.mp4'
        print(self.filename)
        self.FilenamLineEdit.setText(self.filename)

    def save_video(self):
        fps = self.parent.video_fps[0]
        output_width, output_height = self.parent.video_sizes[0] # TODO

        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        output_video = cv2.VideoWriter(str(self.filename), fourcc, fps, (output_width, output_height))

        cutting_pair = []
        for i in range(len(self.parent.cutting_list) // 2):
            cutting_pair.append([self.parent.cutting_list[2*i], self.parent.cutting_list[2*i+1]])
        if len(self.parent.cutting_list) % 2 == 1:
            print(" ** Warning: cutting list has odd count -> ignore the last cutting point **")

        total_output_frame_num = 0
        for s, e in cutting_pair:
            total_output_frame_num += e - s + 1
        print(f"total_output_frame_num: {total_output_frame_num}")

        done_frame_num = 0
        for s, e in cutting_pair:
            for i in range(s, e+1):
                output_video.write(self.parent.frame_all[i])
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
        self.FrameNum_Label.setText("[/]")

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
        self.slider.valueChanged.connect(lambda : self.showImage(self.slider.value()))


        self.playButton.clicked.connect(self.playVideo)
        self.speedUp.blockSignals(False) # TODO - speed up
        self.sliderLeftButton.clicked.connect(self.prevFrame)
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

    def blockFrame1(self, blockBool):
        self.listWidget.blockSignals(blockBool)
        self.listAddButton.setDisabled(blockBool)
        self.listDelButton.setDisabled(blockBool)
        self.listUpButton.setDisabled(blockBool)
        self.listDownButton.setDisabled(blockBool)
        self.setAcceptDrops(not blockBool)

    def blockFrame2(self, blockBool):
        #self.speedUp.blockSignals(blockBool) # TODO - speed up
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

    def addList(self):
        openfiles = QFileDialog.getOpenFileNames(self, 'Open file', './')[0]
        for f in openfiles:
            if f in self.file_list or not is_video(f):
                continue
            self.file_list.append(f)
            self.listWidget.addItem(f)
    def delList(self):
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
            self.slider.setValue(0)
            self.ImageLabel.clear()
            self.blockFrame1(False) # un-block
            self.frame_all = []
            self.video_sizes = []
            self.video_fps = []
            self.video_frame_num = []
            self.video_frame_accum = []
            self.total_frame_num = 0
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
            if cap.isOpened():
                while True:
                    ret, frame = cap.read()
                    if ret:
                        # print(f"{done_idx}/{total_frame_num} in {video_file}")
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        self.frame_all.append(frame)
                        real_frame_num += 1
                        loading_cnt += 1
                        self.listpBar.setValue(loading_cnt)
                    else:
                        break
            if est_frame_num != real_frame_num:
                print(f"Warning: {video_file} has invalid frame: est {est_frame_num} -> real {real_frame_num}")
                self.video_frame_num[i] = real_frame_num
            cap.release()
        self.total_frame_num = sum(self.video_frame_num)
        self.slider.setMaximum(self.total_frame_num - 1)
        self.video_frame_accum = [ sum(self.video_frame_num[:x+1]) for x in range(len(self.video_frame_num)) ]
        assert(loading_cnt == self.total_frame_num)
        self.listpBar.setVisible(False)
        self.showImage(0)

    def playVideo(self):
        self.playMode = not self.playMode
        if self.playMode:
            self.VideoThread = threading.Thread(target=self.playVideoThread, daemon=True)
            self.VideoThread.start() # run playVideoThread
    def prevFrame(self):
        if self.slider.value() > 0:
            self.slider.setValue(self.slider.value() - 1)
    def nextFrame(self):
        if self.slider.value() < len(self.frame_all) - 1:
            self.slider.setValue(self.slider.value() + 1)

    def playVideoThread(self):
        while self.playMode:
            current_frame = self.slider.value()
            if current_frame == len(self.frame_all) - 1:
                self.playMode = False
                break
            self.slider.setValue(current_frame + 1)
            sleep(1 / self.video_fps[0])

    def showImage(self, frame_idx):
        #print(f"[{frame_idx:04d}/{self.total_frame_num-1}]")
        self.showFilename()
        img = self.frame_all[frame_idx]
        qImg = QImage(img.data, img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        p = pixmap.scaled(640, 480, Qt.KeepAspectRatio)
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
        #print(f"[{curr_idx:04d}/{self.total_frame_num}]")
        self.FrameNum_Label.setText(f"[{curr_idx:04d}/{self.total_frame_num-1}]")
        #print([ curr_idx > fnum for fnum in self.video_frame_accum ])
        file_index = [ curr_idx > fnum for fnum in self.video_frame_accum ].index(False)
        # if curr_idx = 10, video_frame_accum = [100, 200, 300] -> [False, False, False] -> file_index = 0
        # if curr_idx = 110, video_frame_accum = [100, 200, 300] -> [True, False, False] -> file_index = 1
        filename = Path(self.file_list[file_index]).name
        file_frame_idx = curr_idx - self.video_frame_accum[file_index-1] if file_index > 0 else curr_idx
        self.FileName_Label.setText(f"{filename}:{file_frame_idx}")

    def setCutter(self):
        cutting_point = self.slider.value()
        self.cutting_list.append(cutting_point)
        self.cutting_list.sort()
        print(self.cutting_list)
        self.update()

    def delCutter(self):
        del_cutting_point = self.slider.value()
        del_cutting_point_candi = -1
        del_cutting_point_idx = 0
        diff = self.total_frame_num
        for i, cutting_point in enumerate(self.cutting_list):
            if abs(del_cutting_point - cutting_point) < diff:
                diff = abs(del_cutting_point - cutting_point)
                del_cutting_point_candi = cutting_point
                del_cutting_point_idx = i

        if del_cutting_point_candi != -1 and diff < 20:
            del self.cutting_list[del_cutting_point_idx]
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

