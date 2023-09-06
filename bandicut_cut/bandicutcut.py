import sys
import typing
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, uic
from pathlib import Path
import cv2
import pdb

#UI파일 연결
#단, UI파일은 Python 코드 파일과 같은 디렉토리에 위치해야한다.
form_class = uic.loadUiType("bandicutcut.ui")[0]

class DonwWindowClass(QDialog) :
    def __init__(self, center_pos=[0, 0]):
        super().__init__()
        self.setWindowTitle("Done !")
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint)
        x, y = int(center_pos[0] - 55), int(center_pos[1] - 25)
        self.setGeometry(x, y, 110, 25)

#화면을 띄우는데 사용되는 Class 선언
class WindowClass(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("bandicutcut")
        self.setAcceptDrops(True)
        self.cButton.clicked.connect(self.cButtonFn)
        self.fileLine.textChanged.connect(lambda : self.pBar.setValue(0))
        self.fileLine.returnPressed.connect(self.cButtonFn)

    def dragEnterEvent(self, event):
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        self.fileLine.setText(files[0])
        # for f in files: # TODO - multiple files

    def cButtonFn(self):
        self.cButton.blockSignals(True)
        self.fileLine.blockSignals(True)
        file_path = Path(self.fileLine.text())
        if not file_path.exists():
            print("파일이 존재하지 않습니다.")
            return False

        #pdb.set_trace()
        # 입력 동영상 파일 열기
        input_video = cv2.VideoCapture(str(file_path))

        # 원본 동영상의 프레임 너비와 높이 가져오기
        frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        total_frames = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 입력 동영상 크기 확인
        if frame_width == 0 or frame_height == 0:
            print(f"Error: {file_path.name} - Invalid size: {frame_width}x{frame_height}")
            return False

        output_width, output_height = frame_width, frame_height

        # 출력 동영상의 코덱 및 FPS 설정 (AVC1 코덱 사용)
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*'avc1')

        output_path = file_path.parent / f"{file_path.stem}_cut{file_path.suffix}"
        print(output_path)
        output_video = cv2.VideoWriter(str(output_path), fourcc, fps, (output_width, output_height))

        tail_frame = self.tailSec.value() * fps
        record_frame = total_frames - tail_frame
        if record_frame < 0:
            print(f"Error: {file_path.name} - Invalid tail sec: {tail_frame}")
            return False

        for i in range(record_frame):
            ret, frame = input_video.read()
            if not ret:
                continue
            # 프레임을 출력 동영상에 쓰기
            output_video.write(frame)

            self.pBar.setValue(int(i/record_frame*100))
        # 자원 해제
        self.pBar.setValue(100)
        input_video.release()
        output_video.release()

        self.doneWindow()
        self.cButton.blockSignals(False)
        self.fileLine.blockSignals(False)

    def doneWindow(self):
        widget = self.geometry()
        x, y = widget.left(), widget.top()
        w, h = widget.width(), widget.height()

        self.done_window = DonwWindowClass([x + w/2, y + h/2])
        self.done_window.exec_()

if __name__ == "__main__" :
    #QApplication : 프로그램을 실행시켜주는 클래스
    app = QApplication(sys.argv)

    #WindowClass의 인스턴스 생성
    myWindow = WindowClass()

    #프로그램 화면을 보여주는 코드
    myWindow.show()

    #프로그램을 이벤트루프로 진입시키는(프로그램을 작동시키는) 코드
    app.exec_()

