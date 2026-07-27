"""
predict_gui.py
----------------
A simple PyQt5 desktop app that loads the trained model (model.pkl) and
predicts a 1-5 star rating for any review text typed in by the user.

Run:
    python predict_gui.py
"""

import sys
import pickle
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel,
                             QTextEdit, QPushButton, QVBoxLayout)
from PyQt5.QtCore import Qt


class ReviewPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.load_model()

    def load_model(self):
        with open('model.pkl', 'rb') as f:
            self.vectorizer, self.model = pickle.load(f)

    def initUI(self):
        self.setWindowTitle('Review Star Predictor')
        self.setGeometry(100, 100, 500, 300)

        self.label = QLabel('Enter your review below:')
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText('Type or paste a review here...')
        self.predict_btn = QPushButton('Predict Stars')
        self.result_label = QLabel('Predicted Stars: ⭐ (waiting)')
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet('font-size: 18px; font-weight: bold;')

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.predict_btn, alignment=Qt.AlignCenter)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.predict_btn.clicked.connect(self.predict_stars)

    def predict_stars(self):
        review = self.text_edit.toPlainText().strip()
        if not review:
            self.result_label.setText('⚠️ Please enter a review.')
            return

        vec = self.vectorizer.transform([review])
        pred = self.model.predict(vec)[0]
        stars = int(round(pred))
        if stars < 1:
            stars = 1
        elif stars > 5:
            stars = 5

        star_emoji = '⭐' * stars + '☆' * (5 - stars)
        self.result_label.setText(f'Predicted Stars: {star_emoji} ({stars}/5)')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ReviewPredictor()
    window.show()
    sys.exit(app.exec_())
