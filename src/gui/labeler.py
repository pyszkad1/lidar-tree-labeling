import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, QGraphicsView, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF

DATA_WIDTH = 1000
DATA_HEIGHT = 128
Y_STRETCH = 4

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QWidget, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint


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
                painter.setPen(QPen(self.penColor, 2, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                rect = QRect(self.startPoint.toPoint(), event.scenePos())
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
                painter.setPen(QPen(Qt.transparent, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setCompositionMode(QPainter.CompositionMode_Clear)
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
        if array.ndim != 3 or array.shape[2] not in [3, 4]:
            raise ValueError("Array must be a 3D array with 3 or 4 channels.")
        height, width, channels = array.shape
        bytesPerLine = channels * width
        format = QImage.Format_RGB888 if channels == 3 else QImage.Format_RGBA8888
        self.backgroundImage = QImage(array.data, width, height, bytesPerLine, format)
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

        # Make buttons square
        button_size = 100  # Define a fixed size for buttons
        self.button1.setFixedSize(button_size, button_size)
        self.button2.setFixedSize(button_size, button_size)
        self.button3.setFixedSize(button_size, button_size)
        self.button4.setFixedSize(button_size, button_size)

        # Add buttons to the layout with spacers
        self.buttons_layout.addWidget(self.button1)
        self.buttons_layout.addWidget(self.button2)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.buttons_layout.addWidget(self.button3)
        self.buttons_layout.addWidget(self.button4)

        self.button1.clicked.connect(self.setFreehandMode)
        self.button2.clicked.connect(self.setRectangleMode)  # Assuming you have a future use for this
        self.button3.clicked.connect(self.setDrawMode)
        self.button4.clicked.connect(self.setEraseMode)

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

    # Example: Create an RGB array and draw it
    # Create a red square
    red_square = np.random.randint(0, 256, (DATA_HEIGHT*Y_STRETCH, DATA_WIDTH, 3), dtype=np.uint8)

    #window.scene.setBackgroundImageFromArray(red_square)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
