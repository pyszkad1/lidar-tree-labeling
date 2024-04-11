import os
from src.gui.image_transformation import *
from src.gui.HistoryQueue import *
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, \
    QGraphicsView, QSpacerItem, QSizePolicy, QFileDialog
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF



DATA_WIDTH = 1024
DATA_HEIGHT = 128




class DrawableGraphicsScene(QGraphicsScene):
    def __init__(self, width, height, parent=None):
        super(DrawableGraphicsScene, self).__init__(parent)
        self.image_width = width
        self.image_width_height = height
        self.backgroundImage = QImage(width, height, QImage.Format_RGB32)
        self.backgroundImage.fill(Qt.transparent)
        self._background_image_numpy = np.zeros((height, width, 3), dtype=np.uint8)

        self.maskImage = QImage(width, height, QImage.Format_RGBA8888)
        self.maskImage.fill(Qt.transparent)
        self._binary_mask = np.zeros((height, width), dtype=np.uint8)

        self.history = HistoryQueue(10)
        self.history.push(self._binary_mask.copy()) # Save the initial mask state

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

    def make_rect(self, p1: QPoint, p2: QPoint):
        topLeftX = min(p1.x(), p2.x())
        topLeftY = min(p1.y(), p2.y())
        bottomRightX = max(p1.x(), p2.x())
        bottomRightY = max(p1.y(), p2.y())
        return QRect(topLeftX, topLeftY, bottomRightX - topLeftX, bottomRightY - topLeftY)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.isDrawing:
            self.isDrawing = False
            if self.drawingMode == 'Rectangle':
                self.clear()
                # Finalize drawing by updating the maskImage with the tempImage
                painter = QPainter(self.maskImage)
                painter.setBrush(QBrush(self.penColor))
                painter.setPen(Qt.NoPen)
                rect = self.make_rect(self.startPoint.toPoint(), event.scenePos().toPoint())
                painter.drawRect(rect)
                painter.end()
                self.tempImage = None
            if self.drawingMode == 'Smart Select':
                smart_select_mask = self.smart_select(int(event.scenePos().x()), int(event.scenePos().y()))
                self.update_mask_image(smart_select_mask)

            self._update_binary_mask_from_image()
            self.saveMaskState() # Save the current mask state to the history queue
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



    def saveMaskState(self):
        self.history.push(self._binary_mask.copy())

    def undoMaskState(self):
        # Call this method in response to Ctrl+Z
        prev_state = self.history.undo()
        if prev_state is not None:
            self.set_mask_image(prev_state)

    def redoMaskState(self):
        # Call this method in response to Ctrl+Y
        next_state = self.history.redo()
        if next_state is not None:
            self.set_mask_image(next_state)



    def is_similar(self, color1, color2, tolerance):
        """Check if two colors are similar based on a given tolerance."""
        return np.linalg.norm(color1 - color2) <= tolerance

    def smart_select(self, x : int, y : int, tolerance=20):
        """Selects a contiguous region with similar colors around (x, y)."""
        height, width, _ = self._background_image_numpy.shape
        visited = np.zeros((height, width), dtype=bool)
        mask = np.zeros((height, width), dtype=bool)

        # Initial color to compare with
        target_color = self._background_image_numpy[y, x, :]
        # Stack for depth-first search
        stack = [(y, x)]

        while stack:
            current_y, current_x = stack.pop()

            # Skip out-of-bounds or already visited pixels
            if current_x < 0 or current_x >= width or current_y < 0 or current_y >= height or visited[
                current_y, current_x]:
                continue

            visited[current_y, current_x] = True  # Mark as visited

            # Check color similarity
            if self.is_similar(self._background_image_numpy[current_y, current_x, :], target_color, tolerance):
                mask[current_y, current_x] = True  # Mark as part of the region

                # Add neighboring pixels to the stack
                stack.extend([(current_y + dy, current_x + dx) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]])

        return mask

    def update_mask_image(self, mask):
        painter = QPainter(self.maskImage)
        painter.setPen(QPen(Qt.yellow, 1))  # Set the pen color and size
        painter.setBrush(QBrush(Qt.yellow))  # Set the brush color

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x]:
                    self._binary_mask = mask
                    painter.drawPoint(x, y)  # Draw a point for every True value in the mask

        painter.end()
        self.update()

    def set_mask_image(self, mask_array: np.ndarray) -> None:
        # mask is a 2D numpy array with 1s where the mask is and 0s elsewhere
        height, width = mask_array.shape

        rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_image[mask_array == 1] = [255, 255, 0, 255]  # Yellow
        rgba_image[mask_array == 0] = [0, 0, 0, 0]  # Transparent

        qimage = QImage(rgba_image.data, width, height, QImage.Format_RGBA8888)
        self.maskImage = qimage
        self._binary_mask = mask_array
        self.update_mask_image(mask_array)

    def _update_binary_mask_from_image(self):
        # Convert the mask image to a binary mask
        ptr = self.maskImage.bits()
        ptr.setsize(self.maskImage.byteCount())
        mask_array = np.array(ptr).reshape(self.maskImage.height(), self.maskImage.width(), 4)
        mask_array = np.any(mask_array != 0, axis=-1).astype(int)[..., np.newaxis].squeeze()
        self._binary_mask = mask_array

        print("Binary mask updated")


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
        self._background_image_numpy = resized_array
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
        self.button5 = QPushButton("Smart Select")

        # Make buttons square
        button_size = 100  # Define a fixed size for buttons
        self.button1.setFixedSize(button_size, button_size)
        self.button2.setFixedSize(button_size, button_size)
        self.button3.setFixedSize(button_size, button_size)
        self.button4.setFixedSize(button_size, button_size)
        self.button5.setFixedSize(button_size, button_size)

        # Add buttons to the layout with spacers
        self.buttons_layout.addWidget(self.button1)
        self.buttons_layout.addWidget(self.button2)
        self.buttons_layout.addWidget(self.button3)
        self.buttons_layout.addWidget(self.button4)
        self.buttons_layout.addWidget(self.button5)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.button1.clicked.connect(self._set_freehand_mode)
        self.button2.clicked.connect(self._set_rectangle_mode)
        self.button3.clicked.connect(self._set_draw_mode)
        self.button4.clicked.connect(self._set_erase_mode)
        self.button5.clicked.connect(self._set_smart_select_mode)

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

    def _set_smart_select_mode(self):
        self.scene.drawingMode = 'Smart Select'
        self._reset_button_styles()
        self._set_active_button_style(self.button5)

    def get_mask_bits(self) -> np.ndarray:
        ptr = self.scene.maskImage.bits()
        ptr.setsize(self.scene.maskImage.byteCount())
        return np.array(ptr).reshape(self.scene.maskImage.height(), self.scene.maskImage.width(), 4)  # 4 for RGBA





