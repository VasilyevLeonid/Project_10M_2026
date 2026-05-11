import sys
from PyQt5 import QtWidgets
from ui_main import MainWin

if __name__ == "__main__":
    # Создаём приложение
    app = QtWidgets.QApplication(sys.argv)

    # Стиль для подсказок
    app.setStyleSheet("""
        QToolTip {
            background-color: #313244;
            color: #cdd6f4;
            border: 1px solid #89b4fa;
        }
    """)

    # Создаём и показываем главное окно
    win = MainWin()
    win.show()

    # Запускаем цикл обработки событий
    sys.exit(app.exec_())