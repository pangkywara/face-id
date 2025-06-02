
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition System")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout
        layout = QVBoxLayout(self.central_widget)
        
        # Add a label
        label = QLabel("Face Recognition System Started")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
