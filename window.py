import sys
import pickle
from PyQt5.QtWidgets import QApplication, QWidget, \
    QTextEdit, QPushButton, QVBoxLayout, QProgressBar
from PyQt5.QtCore import Qt
from models import SpamDetector

class SpamDetectorWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_model()

    def load_model(self):
        with open('spam-det-model.pkl', 'rb') as f:
            self.model = pickle.load(f)
    
    def init_ui(self):
        # vidjetlarni yasash
        main_layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        main_layout.addWidget(self.text_edit)

        self.button = QPushButton('Submit')
        main_layout.addWidget(self.button)
        self.button.clicked.connect(self.on_submit_clicked)

        self.bar_non_spam = QProgressBar()
        self.bar_non_spam.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.bar_non_spam)

        self.bar_spam = QProgressBar()
        self.bar_spam.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.bar_spam)

        self.setLayout(main_layout)
        self.show()
    
    def on_submit_clicked(self):
        text = self.text_edit.toPlainText()
        y = self.model.predict([text])
        y *= 100
        self.bar_non_spam.setValue(int(y[0, 0]))
        self.bar_spam.setValue(int(y[0, 1]))

        self.bar_non_spam.setFormat(f'Non spam with {int(y[0, 0])}%')
        self.bar_spam.setFormat(f'Spam with {int(y[0, 1])}%')

        # print(f"Spam emasligi: {y[0, 0]:.2f}%, Spamligi: {y[0, 1]:.2f}%")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SpamDetectorWindow()
    sys.exit(app.exec_())