import os
from src.gui.image_transformation import *
from src.gui.HistoryQueue import *
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, \
    QGraphicsView, QSpacerItem, QSizePolicy, QFileDialog, QDialog, QLabel, QComboBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QMouseEvent
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
        self.history.push(self._binary_mask.copy())  # Save the initial mask state

        self.isDrawing = False
        self.lastPoint = QPoint()
        self.drawingMode = 'Draw'

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

    @staticmethod
    def makeRect(p1: QPoint, p2: QPoint):
        top_left_x = min(p1.x(), p2.x())
        top_left_y = min(p1.y(), p2.y())
        bottom_right_x = max(p1.x(), p2.x())
        bottom_right_y = max(p1.y(), p2.y())
        return QRect(top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y)

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
                if smart_select_mask is not None:
                    self.update_mask_image(smart_select_mask)

            self._update_binary_mask_from_image()
            self.saveMaskState() # Save the current mask state to the history queue
            self.update()

    def mouseMoveEvent(self, event):
        #print(event.scenePos())
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

    def smart_select(self, x : int, y : int, tolerance=40):
        """Selects a contiguous region with similar colors around (x, y)."""
        height, width, _ = self._background_image_numpy.shape
        visited = np.zeros((height, width), dtype=bool)
        mask = np.zeros((height, width), dtype=bool)

        # Initial color to compare with
        target_color = self._background_image_numpy[y, x, :]
        if target_color[0] == 0:
            return None
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

    def smart_select_2(self, x : int, y : int, tolerance=30):
        """Selects a contiguous region with similar colors around (x, y)."""
        height, width, _ = self._background_image_numpy.shape
        visited = np.zeros((height, width), dtype=bool)
        mask = np.zeros((height, width), dtype=bool)

        # Initial color to compare with
        target_color = self._background_image_numpy[y, x, :]
        # Stack for depth-first search
        stack = [(y, x, target_color)]

        while stack:
            current_y, current_x, current_target_color = stack.pop()

            # Skip out-of-bounds or already visited pixels
            if current_x < 0 or current_x >= width or current_y < 0 or current_y >= height or visited[
                current_y, current_x]:
                continue

            visited[current_y, current_x] = True  # Mark as visited

            # Check color similarity
            if self.is_similar(self._background_image_numpy[current_y, current_x, :], current_target_color, tolerance):
                mask[current_y, current_x] = True  # Mark as part of the region

                # Add neighboring pixels to the stack
                stack.extend(
                    [(current_y + dy, current_x + dx, self._background_image_numpy[current_y, current_x, :]) for dx, dy
                     in [(-1, 0), (1, 0), (0, -1), (0, 1)]])

        return mask


    def update_mask_image(self, mask):
        painter = QPainter(self.maskImage)
        painter.setPen(QPen(Qt.yellow, 1))  # Set the pen color and size
        painter.setBrush(QBrush(Qt.yellow))  # Set the brush color

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] and not self._binary_mask[y, x] and self._background_image_numpy[y, x, 0] != 0:
                    self._binary_mask[y, x] = 1
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



class SizeSelectionDialog(QDialog):
    def __init__(self, title="Select Size", default_size="10", parent=None):
        super(SizeSelectionDialog, self).__init__(parent, Qt.WindowCloseButtonHint)
        self.setWindowTitle(title)
        self.layout = QVBoxLayout(self)
        self.selected_size = int(default_size)  # Ensure this is always defined

        # Create a label
        self.label = QLabel("Select size:", self)
        self.layout.addWidget(self.label)

        # Create a combo box for selecting size
        self.sizeComboBox = QComboBox(self)
        self.sizeComboBox.addItems(["1", "5", "10", "20"])  # Adding predefined sizes as strings
        self.sizeComboBox.setCurrentText(default_size)  # Set default size
        self.layout.addWidget(self.sizeComboBox)

        # Connect selection change to a method
        self.sizeComboBox.currentIndexChanged.connect(self.on_size_selected)

    def on_size_selected(self, index):
        # Directly extract and apply the selected size
        self.selected_size = int(self.sizeComboBox.currentText())
        self.accept()

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super(ZoomableGraphicsView, self).__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale_factor = 1.15
        self.current_scale = 1.0
        self.panning = False
        self.last_mouse_position = QPoint()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        """Zoom into the view."""
        self.scale(self.scale_factor, self.scale_factor)
        self.current_scale *= self.scale_factor

    def zoom_out(self):
        """Zoom out of the view, with a limit to not zoom out past original size."""
        if self.current_scale > 1.0:  # Check if the next zoom out is above the limit
            self.current_scale /= self.scale_factor
            if self.current_scale < 1.0:
                self.current_scale = 1.0
                self.scale(1, 1)
            else:
                self.scale(1 / self.scale_factor, 1 / self.scale_factor)

    def mousePressEvent(self, event: QMouseEvent):
        # print("Horizontal ScrollBar Range:", self.horizontalScrollBar().minimum(), self.horizontalScrollBar().maximum())
        # print("Vertical ScrollBar Range:", self.verticalScrollBar().minimum(), self.verticalScrollBar().maximum())
        if event.button() == Qt.RightButton:
            self.panning = True
            self.last_mouse_position = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)  # Optional: for built-in hand-drag functionality
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.panning:
            # Calculate how much we moved
            delta = event.pos() - self.last_mouse_position
            self.last_mouse_position = event.pos()

            adjusted_delta = delta / self.current_scale * 1.2  # Adjust the delta based on the current scale

            # Update the scrollbars
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - adjusted_delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - adjusted_delta.y())


        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.RightButton:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            self.setDragMode(QGraphicsView.NoDrag)  # Optional: for built-in hand-drag functionality
        else:
            super().mouseReleaseEvent(event)

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
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedSize(self.image_width, self.image_height)  # Canvas size
        self.scene.setSceneRect(0, 0, self.image_width, self.image_height)  # Set the scene size to match the view
        self.main_layout.addWidget(self.view)

        # Create a layout for buttons
        self.buttons_layout = QHBoxLayout()

        # Create buttons

        self.button_rectangle = QPushButton("Rectangle")

        self.button_draw = QPushButton("Draw")
        self.button_draw.clicked.connect(lambda: self.open_size_selection_dialog("Select Brush Size", self.button_draw))
        self.buttons_layout.addWidget(self.button_draw)

        self.button_erase = QPushButton("Erase")
        self.button_erase.clicked.connect(lambda: self.open_size_selection_dialog("Select Eraser Size", self.button_erase))
        self.buttons_layout.addWidget(self.button_erase)

        self.button_smart_select = QPushButton("Smart Select")

        # Make buttons square
        button_size = 100  # Define a fixed size for buttons

        self.button_rectangle.setFixedSize(button_size, button_size)
        self.button_draw.setFixedSize(button_size, button_size)
        self.button_erase.setFixedSize(button_size, button_size)
        self.button_smart_select.setFixedSize(button_size, button_size)

        # Add buttons to the layout with spacers

        self.buttons_layout.addWidget(self.button_rectangle)
        self.buttons_layout.addWidget(self.button_draw)
        self.buttons_layout.addWidget(self.button_erase)
        self.buttons_layout.addWidget(self.button_smart_select)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.button_rectangle.clicked.connect(self._set_rectangle_mode)
        self.button_draw.clicked.connect(self._set_draw_mode)
        self.button_erase.clicked.connect(self._set_erase_mode)
        self.button_smart_select.clicked.connect(self._set_smart_select_mode)

        # Add buttons layout to the main layout
        self.main_layout.addLayout(self.buttons_layout)
        self._reset_button_styles()
        self._set_draw_mode()

    def _reset_button_styles(self):
        # Define a default stylesheet
        default_style = "QPushButton { background-color: None; }"
        self.button_rectangle.setStyleSheet(default_style)
        self.button_draw.setStyleSheet(default_style)
        self.button_erase.setStyleSheet(default_style)
        self.button_smart_select.setStyleSheet(default_style)

    def _set_active_button_style(self, button):
        # Define an active button stylesheet
        active_style = "QPushButton { background-color: grey; }"
        button.setStyleSheet(active_style)

    def _set_rectangle_mode(self):
        self.scene.drawingMode = 'Rectangle'
        self.scene.penColor = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button_rectangle)

    def _set_draw_mode(self):
        self.scene.drawingMode = 'Draw'
        self.scene.penColor = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button_draw)

    def _set_erase_mode(self):
        self.scene.drawingMode = 'Erase'
        self._reset_button_styles()
        self._set_active_button_style(self.button_erase)

    def _set_smart_select_mode(self):
        self.scene.drawingMode = 'Smart Select'
        self._reset_button_styles()
        self._set_active_button_style(self.button_smart_select)

    def open_size_selection_dialog(self, title, button):
        try:
            button_pos = button.mapToGlobal(button.pos())
            dialog = SizeSelectionDialog(title, "10", self)
            dialog.move(button_pos + QPoint(0, button.height()))  # Position it below the button
            if dialog.exec_() == QDialog.Accepted:
                selected_size = dialog.get_selected_size()
                if 'brush' in title.lower():
                    self.set_brush_size(selected_size)
                elif 'eraser' in title.lower():
                    self.set_eraser_size(selected_size)
        except Exception as e:
            print(f"Error opening size selection dialog: {e}")

    def set_brush_size(self, size):
        print(f"Brush size set to: {size}")

    def set_eraser_size(self, size):
        print(f"Eraser size set to: {size}")

    def get_mask_bits(self) -> np.ndarray:
        ptr = self.scene.maskImage.bits()
        ptr.setsize(self.scene.maskImage.byteCount())
        return np.array(ptr).reshape(self.scene.maskImage.height(), self.scene.maskImage.width(), 4)  # 4 for RGBA





