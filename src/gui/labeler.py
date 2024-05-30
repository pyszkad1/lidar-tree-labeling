import os
from src.gui.image_transformation import *
from src.gui.HistoryQueue import *
from src.gui.NNControls import *
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QGraphicsScene, \
    QGraphicsView, QSpacerItem, QSizePolicy, QFileDialog, QDialog, QLabel, QComboBox, QSlider, QMessageBox
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QMouseEvent, QColor
from PyQt5.QtCore import Qt, QPoint, QRect, QRectF, QObject, pyqtSignal, QThread

DATA_WIDTH = 1024
DATA_HEIGHT = 128


class DrawableGraphicsScene(QGraphicsScene):
    def __init__(self, width, height, real_height, parent=None):
        super(DrawableGraphicsScene, self).__init__(parent)
        self.image_width = width
        self.image_height = real_height
        self.backgroundImage = QImage(width, height, QImage.Format_RGB32)
        self.backgroundImage.fill(Qt.transparent)
        self._background_image_numpy = np.zeros((height, width, 3), dtype=np.uint8)
        self._range_array = np.zeros((real_height, width), dtype=np.uint8)

        self.maskImage = QImage(width, real_height, QImage.Format_RGBA8888)
        self.maskImage.fill(Qt.transparent)
        self._binary_mask = np.zeros((height, width), dtype=np.uint8)

        self.history = HistoryQueue(10)
        self.history.push(self._binary_mask.copy())  # Save the initial mask state

        self.temp_history = HistoryQueue(2)

        self.isDrawing = False
        self.lastPoint = QPoint()
        self.drawingMode = 'Draw'

        self.pen_size = 10
        self.eraser_size = 10
        self.current_opacity = 1.0

        self.pen_color = QColor(Qt.yellow)
        self.startPoint = QPoint()
        self.endPoint = QPoint()
        self.tempImage = None  # Temporary image for drawing shapes

    @staticmethod
    def make_rect(p1: QPoint, p2: QPoint):
        top_left_x = min(p1.x(), p2.x())
        top_left_y = min(p1.y(), p2.y())
        bottom_right_x = max(p1.x(), p2.x())
        bottom_right_y = max(p1.y(), p2.y())
        return QRect(top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y)

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
                painter.setBrush(QBrush(self.pen_color))
                painter.setPen(Qt.NoPen)
                rect = self.make_rect(self.startPoint.toPoint(), event.scenePos().toPoint())
                painter.drawRect(rect)
                painter.end()
                self.tempImage = None
            if self.drawingMode == 'Smart Select':

                if event.scenePos().x() > self.image_width or event.scenePos().y() > self.image_height:
                    return
                smart_select_mask = self.smart_select(int(event.scenePos().x()), int(event.scenePos().y()))
                if smart_select_mask is not None:
                    self.update_mask_image(smart_select_mask)

            self._update_binary_mask_from_image()
            self.update_opacity(int(self.current_opacity * 100))
            self.saveMaskState()  # Save the current mask state to the history queue
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and self.isDrawing:
            painter = QPainter(self.maskImage)

            if self.drawingMode == 'Rectangle':
                tempPixmap = QPixmap.fromImage(self.tempImage)
                self.clear()  # Clear the scene
                self.addPixmap(QPixmap.fromImage(self.backgroundImage))  # Redraw the background image
                self.addPixmap(tempPixmap)  # Add the unmodified mask image

                painter = QPainter(tempPixmap)
                painter.setPen(QPen(self.pen_color, 2, Qt.SolidLine, Qt.SquareCap, Qt.BevelJoin))
                rect = QRectF(self.startPoint,
                              event.scenePos())  # Create a rectangle from the start point to the current mouse position
                painter.drawRect(rect)
                painter.end()

                self.addPixmap(tempPixmap)
            elif self.drawingMode == 'Erase':
                painter.setPen(QPen(Qt.transparent, self.eraser_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setCompositionMode(QPainter.CompositionMode_Clear)

                painter.drawLine(self.lastPoint, event.scenePos())

            elif self.drawingMode == 'Draw':
                painter.setPen(QPen(self.pen_color, self.pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.drawLine(self.lastPoint, event.scenePos())
            elif self.drawingMode == 'Smart Select':
                pass
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

    def clear_undo_redo(self):
        self.history = HistoryQueue(10)
        self.history.push(self._binary_mask.copy())

    def is_similar(self, color1, color2, tolerance):
        """Check if two colors are similar based on a given tolerance."""
        return np.linalg.norm(color1 - color2) <= tolerance

    def smart_select(self, x: int, y: int, tolerance=40):
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

    def update_mask_image(self, mask: np.ndarray):
        painter = QPainter(self.maskImage)
        painter.setPen(QPen(Qt.yellow, 1))  # Set the pen color and size
        painter.setBrush(QBrush(Qt.yellow))  # Set the brush color

        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y, x] and not self._binary_mask[y, x] and self._background_image_numpy[y, x, 0] != 0:
                    self._binary_mask[y, x] = 1
                    painter.drawPoint(x, y)  # Draw a point for every True value in the mask

        painter.end()
        if self.current_opacity < 1.0:
            self.update_opacity(int(self.current_opacity * 100))
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

    def clear_mask(self):
        self.maskImage.fill(Qt.transparent)
        self._binary_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        self.update()

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

    def update_opacity(self, value):
        """Update the opacity of the image based on the slider value."""
        # Calculate opacity (0.0 to 1.0)
        opacity = value / 100.0
        self.current_opacity = opacity

        image = self.maskImage
        if image.format() != QImage.Format_ARGB32 and image.format() != QImage.Format_ARGB32_Premultiplied:
            image = image.convertToFormat(QImage.Format_ARGB32)

        for x in range(image.width()):
            for y in range(image.height()):
                if self._binary_mask[y, x] == 0:
                    continue
                color = QColor(image.pixel(x, y))
                color.setAlpha(int(255 * opacity))  # Set alpha based on opacity factor (0.0 to 1.0)
                image.setPixel(x, y, color.rgba())

        self.maskImage = image
        self.update()
        print("Opacity updated")



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
        if event.button() == Qt.RightButton:
            self.panning = True
            self.last_mouse_position = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.panning:
            # Calculate how much we moved
            delta = event.pos() - self.last_mouse_position
            self.last_mouse_position = event.pos()

            adjusted_delta = delta / min(self.current_scale,2) * 1.55  # Adjust the delta based on the current scale

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

    def set_background_image(self, rgb_array: np.ndarray, range_array: np.ndarray) -> None:
        # self.adjustCanvasSize(array.shape[1], array.shape[0])
        self.scene.set_background_image_from_array(rgb_array)
        self.scene._range_array = range_array

    def get_image_qpixmap(self) -> QPixmap:
        return QPixmap.fromImage(self.scene.backgroundImage)

    def get_range_array(self) -> np.ndarray:
        return self.scene._range_array

    def __init__(self, width: int = DATA_WIDTH, height: int = DATA_HEIGHT):
        super().__init__()

        self.image_width = width
        self.image_height = height

        self.NN_controller = NNControls()

        # Main layout for the widget
        self.layout = QVBoxLayout(self)

        self.middle_widget = QWidget()
        self.middle_widget.setLayout(QHBoxLayout())

        # Create the canvas
        self.scene = DrawableGraphicsScene(self.image_width, self.image_height * 4, self.image_height)
        self.view = ZoomableGraphicsView(self.scene)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setFixedSize(self.image_width, self.image_height * 4)  # Canvas size
        self.scene.setSceneRect(0, 0, self.image_width, self.image_height *4)  # Set the scene size to match the view
        self.layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        self.right_widget = self._init_right_widget()

        self.middle_widget.layout().addWidget(self.view)
        self.middle_widget.layout().addWidget(self.right_widget, alignment=Qt.AlignVCenter)

        self.layout.addWidget(self.middle_widget)

        # Create a layout for buttons
        self.buttons_layout = QHBoxLayout()

        # Create buttons

        self.button_rectangle = QPushButton("Rectangle\nSelect")
        self.button_smart_select = QPushButton("Smart\nSelect")

        slider_width = 100
        slider_height = 20

        self.draw_layout = QVBoxLayout()
        self.draw_slider = QSlider(Qt.Horizontal)
        self.draw_slider.setMinimum(1)
        self.draw_slider.setMaximum(20)
        self.draw_slider.setValue(10)  # Default size for drawing
        self.draw_slider.setFixedSize(slider_width, slider_height)
        self.draw_slider.valueChanged.connect(lambda: self.set_size(self.draw_slider.value(), 'draw'))

        self.draw_button_pane = QWidget()

        self.draw_button_pane.setLayout(self.draw_layout)

        self.button_draw = QPushButton("Draw")

        self.draw_layout.addWidget(self.button_draw)
        label = QLabel("Pen Size:")
        self.draw_layout.addWidget(label)
        self.draw_layout.addWidget(self.draw_slider)
        self.draw_button_pane.setMaximumWidth(100)
        self.draw_layout.setContentsMargins(0, 0, 0, 0)

        # Set up the slider and button for erasing
        self.erase_layout = QVBoxLayout()
        self.erase_button_pane = QWidget()
        self.erase_button_pane.setLayout(self.erase_layout)
        self.erase_button_pane.setMaximumWidth(100)
        self.erase_layout.setContentsMargins(0, 0, 0, 0)
        self.erase_slider = QSlider(Qt.Horizontal)
        self.erase_slider.setMinimum(1)
        self.erase_slider.setMaximum(20)
        self.erase_slider.setValue(10)  # Default size for erasing
        self.erase_slider.setFixedSize(slider_width, slider_height)
        self.erase_slider.valueChanged.connect(lambda: self.set_size(self.erase_slider.value(), 'erase'))

        self.button_erase = QPushButton("Erase")

        self.erase_layout.addWidget(self.button_erase)
        self.erase_layout.addWidget(QLabel("Eraser Size:"))
        self.erase_layout.addWidget(self.erase_slider)
        self.layout.addLayout(self.erase_layout)

        self.opacity_layout = QVBoxLayout()
        self.opacity_pane = QWidget()
        self.opacity_pane.setLayout(self.opacity_layout)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setTickInterval(10)
        self.opacity_slider.setFixedSize(slider_width, slider_height)
        self.opacity_slider.sliderReleased.connect(lambda: self.scene.update_opacity(self.opacity_slider.value()))
        self.opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_layout.addWidget(self.opacity_slider)
        self.layout.addLayout(self.opacity_layout)


        # Make buttons square
        button_size = 100  # Define a fixed size for buttons

        self.button_rectangle.setFixedSize(button_size, button_size)
        self.button_draw.setFixedSize(button_size, button_size)
        self.button_erase.setFixedSize(button_size, button_size)
        self.button_smart_select.setFixedSize(button_size, button_size)

        # Add buttons to the layout with spacers

        self.buttons_layout.addWidget(self.draw_button_pane, alignment=Qt.AlignTop)
        self.buttons_layout.addWidget(self.button_rectangle, alignment=Qt.AlignTop)
        self.buttons_layout.addWidget(self.erase_button_pane, alignment=Qt.AlignTop)
        self.buttons_layout.addWidget(self.button_smart_select, alignment=Qt.AlignTop)
        self.buttons_layout.addWidget(self.opacity_pane, alignment=Qt.AlignTop)
        self.buttons_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        widget = QWidget()
        widget.setLayout(self.buttons_layout)
        widget.setFixedHeight(200)

        self.button_rectangle.clicked.connect(self._set_rectangle_mode)
        self.button_draw.clicked.connect(self._set_draw_mode)
        self.button_erase.clicked.connect(self._set_erase_mode)
        self.button_smart_select.clicked.connect(self._set_smart_select_mode)

        # Add buttons layout to the main layout
        self.layout.addWidget(widget)

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
        self.scene.pen_color = Qt.yellow
        self._reset_button_styles()
        self._set_active_button_style(self.button_rectangle)

    def _set_draw_mode(self):
        self.scene.drawingMode = 'Draw'
        self.scene.pen_color = Qt.yellow
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

    def set_size(self, size, tool_context):
        if tool_context == 'draw':
            self.scene.pen_size = size
            self._set_draw_mode()
        elif tool_context == 'erase':
            self.scene.eraser_size = size
            self._set_erase_mode()


    def get_mask_bits(self) -> np.ndarray:
        ptr = self.scene.maskImage.bits()
        ptr.setsize(self.scene.maskImage.byteCount())
        return np.array(ptr).reshape(self.scene.maskImage.height(), self.scene.maskImage.width(), 4)  # 4 for RGBA

    def _init_right_widget(self):
        default_style = "QPushButton { background-color: None; }"
        button_size = 100


        right_widget = QWidget()
        right_pane_layout = QVBoxLayout()
        learn_button = QPushButton("Learn")
        learn_button.clicked.connect(self._start_training)
        learn_button.setStyleSheet(default_style)
        learn_button.setFixedSize(button_size, button_size)

        predict_mask_button = QPushButton("\nPredict\nMask")
        predict_mask_button.setStyleSheet(default_style)
        predict_mask_button.clicked.connect(lambda: self.NN_controller.test_UNet(self.scene._range_array))
        predict_mask_button.setFixedSize(button_size, button_size)

        use_mask_button = QPushButton("Use\nPredicted\nMask")
        use_mask_button.clicked.connect(self._use_predicted_mask)
        use_mask_button.setStyleSheet(default_style)
        use_mask_button.setFixedSize(button_size, button_size)

        # Add spacer to push buttons to the right
        right_pane_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Minimum))

        right_pane_layout.addWidget(learn_button, alignment=Qt.AlignRight)
        right_pane_layout.addWidget(predict_mask_button,alignment=Qt.AlignRight)
        right_pane_layout.addWidget(use_mask_button,alignment=Qt.AlignRight)

        right_widget.setLayout(right_pane_layout)
        return right_widget


    def _use_predicted_mask(self):
        mask = self.NN_controller.current_predictions
        self.scene.set_mask_image(mask)
        self.scene.saveMaskState()

    def _start_training(self):
        if self.NN_controller.is_learning:
            QMessageBox.information(self, "Training in progress", "Training is already in progress.")
            return
        self.NN_controller.is_learning = True
        self.thread = QThread()
        self.worker = TrainingWorker(self.NN_controller)

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.train_model)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.stopped_learning)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self._handle_error)

        self.thread.start()

    def stopped_learning(self):
        self.NN_controller.is_learning = False
        QMessageBox.information(self, "Training complete", "Training is complete.")

    def _handle_error(self, e):
        QMessageBox.critical(self, "Error", str(e))
        self.thread.quit()



class TrainingWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(Exception)

    def __init__(self, NNControls):
        super().__init__()
        self.NNControls = NNControls

    def train_model(self):
        try:
            self.NNControls.learn_UNet()
            self.finished.emit()
        except Exception as e:
            self.error.emit(e)




