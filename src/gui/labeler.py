import os
from src.gui.image_transformation import *
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, \
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
        self.backgroundImage.fill(Qt.transparent)
        self.maskImage = QImage(width, height, QImage.Format_RGBA8888)
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
                self.clear()
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
        painter.drawImage(rect, self.maskImage, rect)

        # painter.drawImage(0, 0, self.maskImage)

    def getMaskArray(self):
        mask = np.array(self.maskImage.convertToFormat(QImage.Format_Grayscale8).bits()).reshape(
            (self._height, self._width))
        mask = np.where(mask > 0, 1, 0)
        return mask

    def set_background_image_from_array(self, array: np.ndarray):
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


class Labeler(QWidget):

    def get_drawn_mask(self) -> QPixmap:
        return QPixmap.fromImage(self.scene.maskImage)

    def set_background_image(self, array: np.ndarray) -> None:
        # self.adjustCanvasSize(array.shape[1], array.shape[0])
        self.scene.set_background_image_from_array(array)

    def get_image_qpixmap(self) -> QPixmap:
        return QPixmap.fromImage(self.scene.backgroundImage)

    def __init__(self, width: int = DATA_WIDTH, height: int = DATA_HEIGHT):
        super().__init__()

        self.image_width = width
        self.image_height = height

        # Main layout for the widget
        self.main_layout = QVBoxLayout(self)

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
        self.buttons_layout.addWidget(self.button3)
        self.buttons_layout.addWidget(self.button4)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.button1.clicked.connect(self._set_freehand_mode)
        self.button2.clicked.connect(self._set_rectangle_mode)
        self.button3.clicked.connect(self._set_draw_mode)
        self.button4.clicked.connect(self._set_erase_mode)

        # Add buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)
        self._reset_button_styles()

    def _reset_button_styles(self):
        # Define a default stylesheet
        defaultStyle = "QPushButton { background-color: None; }"
        self.button1.setStyleSheet(defaultStyle)
        self.button2.setStyleSheet(defaultStyle)
        self.button3.setStyleSheet(defaultStyle)
        self.button4.setStyleSheet(defaultStyle)

    def _set_active_button_style(self, button):
        # Define an active button stylesheet
        activeStyle = "QPushButton { background-color: grey; }"
        button.setStyleSheet(activeStyle)

    def _set_freehand_mode(self):
        self.scene.drawingMode = 'Freehand'
        self.scene.penColor = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button1)

    def _set_rectangle_mode(self):
        self.scene.drawingMode = 'Rectangle'
        self.scene.penColor = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button2)

    def _set_draw_mode(self):
        self.scene.drawingMode = 'Draw'
        self.scene.penColor = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button3)

    def _set_erase_mode(self):
        self.scene.drawingMode = 'Erase'
        self._reset_button_styles()
        self._set_active_button_style(self.button4)

    def get_mask_bits(self) -> np.ndarray:
        ptr = self.scene.maskImage.bits()
        ptr.setsize(self.scene.maskImage.byteCount())
        return np.array(ptr).reshape(self.scene.maskImage.height(), self.scene.maskImage.width(), 4)  # 4 for RGBA

    def set_mask_image(self, mask_array: np.ndarray) -> None:

        height, width = mask_array.shape

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[mask_array == 1] = [255, 255, 0, 255]  # Yellow
        rgba_image[mask_array == 0] = [0, 0, 0, 0]  # Transparent

        qimage = QImage(rgba_image.data, width, height, QImage.Format_RGBA8888)
        self.scene.maskImage = qimage
        self.scene.update()



