import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QWindow, QPixmap, QPainter
from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QWidget, QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy, \
    QFileDialog

from src.gui.image_transformation import transform_file
from src.gui.labeler import Labeler


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Labeler")
        self.setGeometry(100, 100, 1500, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        top_pane = self._init_top_widget()
        main_layout.addWidget(top_pane)

        self.labeler = Labeler()

        main_layout.addWidget(self.labeler)

        self.show()

    def _init_top_widget(self, ):
        top_pane = QWidget()
        top_pane_layout = QHBoxLayout(top_pane)
        # Add a button to open a file
        self.open_button = QPushButton("Open")
        self.open_button.clicked.connect(self.open_image)
        top_pane_layout.addWidget(self.open_button)
        # Add a button to save the file
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_files)
        top_pane_layout.addWidget(self.save_button)
        self.load_mask_button = QPushButton("Load Mask From File")
        self.load_mask_button.clicked.connect(self.open_mask_file)
        top_pane_layout.addWidget(self.load_mask_button)


        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        top_pane_layout.addSpacerItem(spacer)

        return top_pane



    def _prompt_for_file(self, namefilter="PCD Files (*.pcd)"):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter(namefilter)
        file_dialog.setDirectory(r"C:\Users\Adam\Desktop\school\Bc_projekt\labeling\data")

        res = file_dialog.exec_()
        if not res:
            return None
        value = file_dialog.selectedFiles()[0]
        return value

    def open_image(self):
        filename = self._prompt_for_file()
        if not filename:
            return
        try:
            rgb_array, range_array = transform_file(filename)
            self.labeler.set_background_image(rgb_array, range_array)
            self.labeler.scene.clear_mask()
        except Exception as e:
            print("Error:", e)


    def open_mask_file(self):
        filename = self._prompt_for_file(namefilter="Numpy Files (*.*)")
        if not filename:
            return
        try:
            mask_array = np.load(filename, allow_pickle=True)
            self.labeler.scene.set_mask_image(mask_array)
        except Exception as e:
            print("Error:", e)

    def save_files(self):
        filename = self._prompt_for_save_name()

        if not filename:
            return

        base_filename, _ = os.path.splitext(filename)  # Removes extension if present

        try:
            # Combine backgroundImage and maskImage
            mask_map = self.labeler.get_drawn_mask()
            background_image = self.labeler.get_image_qpixmap()
            range_array = self.labeler.get_range_array()

            combined_image = self.combine_images(background_image, mask_map)

            # Save the combined image
            combined_image_filename = f"{base_filename}.png"
            combined_image.save(combined_image_filename)
            print("Combined image saved successfully:", combined_image_filename)

            # Convert maskImage to binary mask and save
            binary_mask = self._get_mask_as_binary()
            binary_mask_filename = f"{base_filename}.bin"
            np.save(binary_mask_filename, binary_mask)
            print("Binary mask saved successfully:", binary_mask_filename)

            # Save range array
            range_array_filename = f"{base_filename}.npy"
            np.save(range_array_filename, range_array)
            print("Range array saved successfully:", range_array_filename)

        except Exception as e:
            print("Failed to save files:", e)

    def _prompt_for_save_name(self):
        options = QFileDialog.Options()
        file_dialog = QFileDialog()
        file_dialog.setDirectory(r"C:\Users\Adam\Desktop\school\Bc_projekt\labeling\data\labeled")


        filename, _ = QFileDialog.getSaveFileName(self, "Save Files", "",
                                                  "All Files (*)", options=options)
        return filename

    def _get_mask_as_binary(self):
        # Assuming maskImage is a QImage, convert it to a numpy array first
        arr = self.labeler.get_mask_bits()

        # Create a binary mask: 1 where the pixel is yellow, 0 elsewhere
        # saved as 2d array
        binary_mask = np.any(arr != 0, axis=-1).astype(int)[..., np.newaxis].squeeze()

        return binary_mask

    def combine_images(self, background_pixmap: QPixmap, mask_pixmap: QPixmap):
        painter = QPainter(background_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)  # Ensure mask overlays background
        painter.drawPixmap(background_pixmap.rect(), mask_pixmap)
        painter.end()
        return background_pixmap.toImage()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and (event.modifiers() & Qt.ControlModifier):
            self.save_files()
        if event.key() == Qt.Key_Z and (event.modifiers() & Qt.ControlModifier):
            self.labeler.scene.undoMaskState()
        if event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier):
            self.labeler.scene.redoMaskState()




