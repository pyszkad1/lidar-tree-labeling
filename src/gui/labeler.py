import sys
from image_transformation import *
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, \
    QGraphicsView, QSpacerItem, QSizePolicy, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF

DATA_WIDTH = 1024
DATA_HEIGHT = 128
Y_STRETCH = 4


def make_rect(p1: QPoint, p2: QPoint):
    topLeftX = min(p1.x(), p2.x())
    topLeftY = min(p1.y(), p2.y())
    bottomRightX = max(p1.x(), p2.x())
    bottomRightY = max(p1.y(), p2.y())
    return QRect(topLeftX, topLeftY, bottomRightX - topLeftX, bottomRightY - topLeftY)


class DrawableGraphicsScene(QGraphicsScene):
    def __init__(self, width, height, parent=None):
        super(DrawableGraphicsScene, self).__init__(parent)
        self.image_width = width
        self.image_width_height = height
        self.backgroundImage = QImage(width, height, QImage.Format_RGB32)
        self.backgroundImage.fill(Qt.red)
        self.maskImage = QImage(width, height, QImage.Format_ARGB32)
        self.maskImage.fill(Qt.transparent)
        self.isDrawing = False
        self.lastPoint = QPoint()
        self.drawingMode = 'Freehand'  # Default drawing mode
        self.penColor = Qt.yellow  # Default pen color for drawin
        self.startPoint = QPoint()
        self.endPoint = QPoint()
        self.tempImage = None  # Temporary image for drawing shapes

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        #file_dialog.setNameFilter("Text Files (*.txt)")
        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            try:
                # with open(file_name, 'r') as file:
                #     content = file.read()
                #     # Convert each line into a float and reshape the result into a 2D array with one column
                #     array = np.array([float(line) for line in content.split('\n') if line], ndmin=2).reshape(-1, 1)
                #     rgb_array = self.convertArrayToRGB(array)
                rgb_array = transform_file(file_name)
                self.setBackgroundImageFromArray(rgb_array)
            except Exception as e:
                print("Error:", e)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.isDrawing = True
            self.lastPoint = event.scenePos()
            self.startPoint = event.scenePos()
            if self.drawingMode == 'Rectangle':
                self.tempImage = self.maskImage.copy()  # Copy the current mask image



    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isDrawing:
            self.isDrawing = False
            if self.drawingMode == 'Rectangle':
                # Finalize drawing by updating the maskImage with the tempImage
                painter = QPainter(self.maskImage)
                painter.setBrush(QBrush(self.penColor))
                painter.setPen(Qt.NoPen)
                rect = make_rect(self.startPoint.toPoint(), event.scenePos().toPoint())
                painter.drawRect(rect)
                painter.end()
                self.tempImage = None
            self.update()


    def mouseMoveEvent(self, event):
        print(event.scenePos())
        if event.buttons() & Qt.LeftButton and self.isDrawing:
            painter = QPainter(self.maskImage)

            if self.drawingMode == 'Rectangle':
                tempPixmap = QPixmap.fromImage(self.tempImage)
                self.clear()  # Clear the scene
                self.addPixmap(QPixmap.fromImage(self.backgroundImage))  # Redraw the background image
                self.addPixmap(tempPixmap)  # Add the unmodified mask image

                painter = QPainter(tempPixmap)
                painter.setPen(QPen(self.penColor, 2, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                rect = QRectF(self.startPoint,
                              event.scenePos())  # Create a rectangle from the start point to the current mouse position
                painter.drawRect(rect)
                painter.end()

                self.addPixmap(tempPixmap)
            elif self.drawingMode == 'Erase':
                print("I am erasing")

                eraserSize = 10  # Adjust the eraser size as needed
                painter.setPen(QPen(Qt.transparent, eraserSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setCompositionMode(QPainter.CompositionMode_Clear)

                painter.drawLine(self.lastPoint, event.scenePos())

            else:
                painter.setPen(QPen(self.penColor, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.lastPoint, event.scenePos())
            self.lastPoint = event.scenePos()
            self.update()
            del painter


    def drawBackground(self, painter, rect):
        painter.drawImage(rect, self.backgroundImage, rect)

    def drawForeground(self, painter, rect):
        painter.drawImage(0, 0, self.maskImage)

    def getMaskArray(self):
        mask = np.array(self.maskImage.convertToFormat(QImage.Format_Grayscale8).bits()).reshape((self._height, self._width))
        mask = np.where(mask > 0, 1, 0)
        return mask

    def setBackgroundImageFromArray(self, array):
        # Ensure the array is in the correct 3D shape (height, width, channels)
        if array.ndim != 3 or array.shape[2] not in [3, 4]:
            raise ValueError("Array must be a 3D array with 3 or 4 channels.")


        resized_array = np.resize(array, (DATA_HEIGHT, DATA_WIDTH, array.shape[2]))

        # Calculate bytesPerLine for the QImage
        height, width, channels = resized_array.shape
        bytesPerLine = channels * width

        # Choose format based on the number of channels
        format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888

        # Create QImage from the resized array
        self.backgroundImage = QImage(resized_array.data, width, height, bytesPerLine, format)
        self.update()

class Window(QMainWindow):

    def __init__(self, width, height):
        super().__init__()

        self.image_width = width
        self.image_height = height

        # Set main window properties
        self.setWindowTitle("Adams Labeler")
        self.setGeometry(100, 100, width + 20 , height + 120)  # window size

        # Create a widget to hold everything
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout for the widget
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create the canvas
        self.scene = DrawableGraphicsScene(self.image_width, self.image_height)
        self.view = QGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedSize(self.image_width, self.image_height)  # Canvas size
        self.scene.setSceneRect(0, 0, self.image_width, self.image_height)  # Set the scene size to match the view
        self.main_layout.addWidget(self.view)

        # Create a layout for buttons
        self.buttons_layout = QHBoxLayout()

        # Create buttons
        self.button1 = QPushButton("Freehand")
        self.button2 = QPushButton("Rectangle")
        self.button3 = QPushButton("Draw")
        self.button4 = QPushButton("Erase")
        self.open_button = QPushButton("Open File")

        # Make buttons square
        button_size = 100  # Define a fixed size for buttons
        self.button1.setFixedSize(button_size, button_size)
        self.button2.setFixedSize(button_size, button_size)
        self.button3.setFixedSize(button_size, button_size)
        self.button4.setFixedSize(button_size, button_size)
        self.open_button.setFixedSize(button_size, button_size)

        # Add buttons to the layout with spacers
        self.buttons_layout.addWidget(self.button1)
        self.buttons_layout.addWidget(self.button2)
        self.buttons_layout.addWidget(self.button3)
        self.buttons_layout.addWidget(self.button4)
        self.buttons_layout.addWidget(self.open_button)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.button1.clicked.connect(self.setFreehandMode)
        self.button2.clicked.connect(self.setRectangleMode)
        self.button3.clicked.connect(self.setDrawMode)
        self.button4.clicked.connect(self.setEraseMode)
        self.open_button.clicked.connect(self.scene.openFile)

        # Add buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)
        self.resetButtonStyles()

    def resetButtonStyles(self):
        # Define a default stylesheet
        defaultStyle = "QPushButton { background-color: None; }"
        self.button1.setStyleSheet(defaultStyle)
        self.button2.setStyleSheet(defaultStyle)
        self.button3.setStyleSheet(defaultStyle)
        self.button4.setStyleSheet(defaultStyle)
        self.open_button.setStyleSheet(defaultStyle)

    def setActiveButtonStyle(self, button):
        # Define an active button stylesheet
        activeStyle = "QPushButton { background-color: grey; }"
        button.setStyleSheet(activeStyle)

    def setFreehandMode(self):
        self.scene.drawingMode = 'Freehand'
        self.scene.penColor = Qt.yellow
        self.resetButtonStyles()
        self.setActiveButtonStyle(self.button1)

    def setRectangleMode(self):
        self.scene.drawingMode = 'Rectangle'
        self.scene.penColor = Qt.yellow
        self.resetButtonStyles()
        self.setActiveButtonStyle(self.button2)

    def setDrawMode(self):
        self.scene.drawingMode = 'Draw'
        self.scene.penColor = Qt.yellow
        self.resetButtonStyles()
        self.setActiveButtonStyle(self.button3)

    def setEraseMode(self):
        self.scene.drawingMode = 'Erase'
        self.resetButtonStyles()
        self.setActiveButtonStyle(self.button4)

def main():
    app = QApplication(sys.argv)
    window = Window(DATA_WIDTH, DATA_HEIGHT*Y_STRETCH)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
