from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget,
    QFileDialog, QLineEdit, QMessageBox
)
import os
import sys
from parametrization import db_to_volts, process_waveforms

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waveform Parametrization Tool")
        self.setGeometry(100, 100, 600, 400)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Input folder
        self.input_label = QLabel("Input Folder: Not Selected")
        self.input_button = QPushButton("Select Input Folder")
        self.input_button.clicked.connect(self.select_input_folder)

        # Output folder
        self.output_label = QLabel("Output Folder: Not Selected")
        self.output_button = QPushButton("Select Output Folder")
        self.output_button.clicked.connect(self.select_output_folder)

        # Threshold input
        self.threshold_label = QLabel("Threshold (dB):")
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("Enter threshold in dB")

        # Process button
        self.process_button = QPushButton("Process Waveforms")
        self.process_button.clicked.connect(self.process_waveforms)

        # Add widgets to layout
        layout.addWidget(self.input_label)
        layout.addWidget(self.input_button)
        layout.addWidget(self.output_label)
        layout.addWidget(self.output_button)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_input)
        layout.addWidget(self.process_button)

        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_label.setText(f"Input Folder: {folder}")
            self.input_folder = folder

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_label.setText(f"Output Folder: {folder}")
            self.output_folder = folder

    def process_waveforms(self):
        try:
            input_folder = getattr(self, 'input_folder', None)
            output_folder = getattr(self, 'output_folder', None)
            threshold_db = self.threshold_input.text()

            # Validate input
            if not input_folder or not output_folder:
                QMessageBox.critical(self, "Error", "Please select both input and output folders.")
                return
            if not threshold_db:
                QMessageBox.critical(self, "Error", "Please enter a valid threshold in dB.")
                return

            # Convert threshold and process
            threshold_volts = db_to_volts(float(threshold_db))
            process_waveforms(input_folder, output_folder, threshold_volts)
            QMessageBox.information(self, "Success", "Waveform processing completed.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
