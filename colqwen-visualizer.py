
from model_functions import generate_document_and_query, VisualizedPage
from PyQt6.QtCore import (
    QObject,
    QRunnable,
    QSize, 
    Qt,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor
from PyQt6.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QPushButton, 
    QScrollArea, 
    QHBoxLayout, 
    QVBoxLayout, 
    QLabel, 
    QFrame, 
    QLineEdit, 
    QSizePolicy, 
    QDialog, 
    QFileDialog, 
    QDialogButtonBox,
    QWidget,
    QProgressBar,
)
from pyqtwaitingspinner import WaitingSpinner, SpinnerParameters

# Only needed for access to command line arguments
import faulthandler
import sys
import traceback

class WorkerSignals(QObject):
    """Signals from a running worker thread.

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc())

    result
        object data returned from processing, anything

    progress
        float indicating % progress
    """

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(float)

class Worker(QRunnable):
    """Worker thread.

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread.
                     Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function
    """

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        # Add the callback to our kwargs
        self.kwargs["progress_callback"] = self.signals.progress

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()

# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        faulthandler.enable()
        
        self.document_path = "<Not Selected>"
        self.query = "<None>"

        # window properties
        self.setWindowTitle("ColQwen Visualizer")
        self.setFixedSize(QSize(1200, 800))

        # central widget layout
        central_widget = QWidget()
        central_widget.setLayout(QVBoxLayout())

        # information bar
        top_bar_widget = QWidget()
        top_bar_widget.setLayout(QHBoxLayout())
        top_bar_widget.layout().setContentsMargins(1, 1, 1, 1)
        top_bar_widget.layout().setSpacing(5)
        central_widget.layout().addWidget(top_bar_widget)

        # document and query selection labels area
        selection_labels_area = QWidget()
        selection_labels_area.setLayout(QVBoxLayout())
        selection_labels_area.layout().setContentsMargins(1, 1, 1, 1)
        selection_labels_area.layout().setSpacing(5)
        top_bar_widget.layout().addWidget(selection_labels_area)

        #pdf selection area
        pdf_selection_area = QWidget()
        pdf_selection_area.setLayout(QHBoxLayout())
        pdf_selection_area.layout().addWidget(QLabel("PDF Document Path:"))
        pdf_selection = QLineEdit(self.document_path)
        pdf_selection.setReadOnly(True)
        self.pdf_selection = pdf_selection  # Store reference to the QLineEdit for later use

        pdf_selection_area.layout().addWidget(pdf_selection)
        pdf_selection_area.layout().setContentsMargins(1, 1, 1, 1)
        pdf_selection_area.layout().setSpacing(5)
        selection_labels_area.layout().addWidget(pdf_selection_area)

        #query selection area
        query_selection_area = QWidget()
        query_selection_area.setLayout(QHBoxLayout())
        query_selection_area.layout().addWidget(QLabel("Query:"))
        query_selection = QLineEdit(self.query)
        query_selection.setReadOnly(True)
        self.query_selection = query_selection  # Store reference to the QLineEdit for later use

        query_selection_area.layout().addWidget(query_selection)
        query_selection_area.layout().setContentsMargins(1, 1, 1, 1)
        query_selection_area.layout().setSpacing(5)
        selection_labels_area.layout().addWidget(query_selection_area)

        select_button = QPushButton("Change Document/Query")
        select_button.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred))
        select_button.clicked.connect(self.select_button_clicked)

        top_bar_widget.layout().addWidget(select_button)

        scroll_area = QScrollArea()
        pdf_box = QFrame(scroll_area)
        pdf_box.setLayout(QVBoxLayout())
        pdf_box.layout().setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(pdf_box)
        pdf_box.layout().addWidget(QLabel("No pdf loaded."))
        self.pdf_box = pdf_box  # Store reference to the pdf_box for later use

        central_widget.layout().addWidget(scroll_area)

        self.spinner = WaitingSpinner(
            central_widget,
            spinner_parameters=SpinnerParameters(
                center_on_parent=True,
                disable_parent_when_spinning=True,
                window_modality=Qt.WindowModality.ApplicationModal,
            )
        )

        # Set the central widget of the Window.
        self.setCentralWidget(central_widget)
        self.central_widget = central_widget  # Store reference to the central widget for later use

        self.threadpool = QThreadPool()
        thread_count = self.threadpool.maxThreadCount()
        print(f"Multithreading with maximum {thread_count} threads")

    def select_button_clicked(self):
        # This function will be called when the button is clicked.
        # You can implement the logic to select a document or query here.
        dialog = DocumentQuerySelectionDialog(self, self.document_path, self.query)
        if dialog.exec():
            # If the dialog was accepted, update the document path and query.
            self.update_document_and_query(dialog.pdf_selection.text(), dialog.query_selection.text())
        pass

    def update_document_and_query(self, document_path, query):
        # This function is used to update the document path and query, performing
        # necessary calculations to use colqwen to process the pdf documents, query it, and display
        # the results in the UI
        print(f"Selected document path: {document_path}")
        print(f"Selected query: {query}")
        self.document_path = document_path
        self.query = query
        self.pdf_selection.setText(document_path)
        self.query_selection.setText(query)

        self.spinner.start()
        self.progress_bar = QProgressBar(self.central_widget)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setGeometry(425, 400, 400, 100)
        self.progress_bar.show()

        worker = Worker(
            generate_document_and_query,
            self.document_path,
            self.query,
        )
        worker.signals.result.connect(self.display_images)
        worker.signals.finished.connect(self.finish_fn)
        worker.signals.progress.connect(self.progress_fn)
        self.threadpool.start(worker)

    def finish_fn(self):
        print("Finished processing.")
        self.spinner.stop()
        self.progress_bar.deleteLater()

    def display_images(self, visualized_pages):
        # Clear the existing layout in pdf_box
        for i in reversed(range(self.pdf_box.layout().count())): 
            widget = self.pdf_box.layout().itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        if visualized_pages is None:
            print("No images to display.")
            self.pdf_box.layout().addWidget(QLabel("No images to display."))
            return
        
        # This function is used to display the images in the UI.
        # You can implement the logic to display images here.
        print(f"Displaying {len(visualized_pages)} images.")

        has_relevancy_scores = visualized_pages[0].is_highlighted()
        if has_relevancy_scores:
            # If relevancy scores are provided, sort by them and get min/max scores
            visualized_pages = sorted(visualized_pages, key=lambda p: p.relevancy_score, reverse=True)

            # Get min and max relevancy score
            min_score = min(visualized_pages, key=lambda p: p.relevancy_score).relevancy_score
            max_score = max(visualized_pages, key=lambda p: p.relevancy_score).relevancy_score

        # Display new images
        for i, visualized_page in enumerate(visualized_pages):
            row = QWidget()
            row.setLayout(QHBoxLayout())
            label = QLabel()
            label.setPixmap(QPixmap.fromImage(QImage(visualized_page.image_path)).scaledToWidth(1000))
            row.layout().addWidget(label)

            # Setup annotation area for the page
            annotation_area = QWidget()
            annotation_area.setLayout(QVBoxLayout())
            # Add relevancy score if available
            if has_relevancy_scores:
                score_label = QLabel(f"{visualized_page.relevancy_score}")
                score_label.setAutoFillBackground(True)
                (background, text) = get_relevancy_color_tags_from_percentage((visualized_page.relevancy_score - min_score)/(max_score - min_score))
                palette = score_label.palette()
                palette.setColor(QPalette.ColorRole.Window, QColor(background))
                palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
                score_label.setPalette(palette)

                annotation_area.layout().addWidget(score_label)
            # Add page num label
            page_num_label = QLabel(f"Page {visualized_page.page_num}")
            page_num_label.setFixedHeight(40)
            annotation_area.layout().addWidget(page_num_label)
            row.layout().addWidget(annotation_area)

            self.pdf_box.layout().addWidget(row)
        
        self.pdf_box.update()
    
    def progress_fn(self, n):
        print(f"Progress: {n * 100:.2f}%")
        self.progress_bar.setValue(int(n * 100))

class DocumentQuerySelectionDialog(QDialog):
    def __init__(self, parent=None, document_path="<Not Selected>", query="<None>"):
        # This dialog can be used to select a document and query.
        super().__init__(parent)
        self.setWindowTitle("Document/Query Selection")
        self.setFixedSize(QSize(800, 85))

        # document and query selection labels area
        central_widget = QFrame()
        central_widget.setLayout(QVBoxLayout())
        central_widget.layout().setContentsMargins(1, 1, 1, 1)
        central_widget.layout().setSpacing(5)

        #pdf selection area
        pdf_selection_area = QFrame()
        pdf_selection_area.setLayout(QHBoxLayout())
        pdf_selection_area.layout().addWidget(QLabel("PDF Document Path:"))
        pdf_selection = QLineEdit(document_path)
        self.pdf_selection = pdf_selection  # Store reference to the QLineEdit for later use
        pdf_selection_area.layout().addWidget(pdf_selection)
        pdf_selection_button = QPushButton("Browse")
        pdf_selection_button.clicked.connect(self.browse_button_clicked)  # Connect to the browse button
        pdf_selection_area.layout().addWidget(pdf_selection_button)

        pdf_selection_area.layout().setContentsMargins(1, 1, 1, 1)
        pdf_selection_area.layout().setSpacing(5)
        central_widget.layout().addWidget(pdf_selection_area)

        #query selection area
        query_selection_area = QFrame()
        query_selection_area.setLayout(QHBoxLayout())
        query_selection_area.layout().addWidget(QLabel("Query:"))
        query_selection = QLineEdit(query)
        self.query_selection = query_selection

        query_selection_area.layout().addWidget(query_selection)
        query_selection_area.layout().setContentsMargins(1, 1, 1, 1)
        query_selection_area.layout().setSpacing(5)
        central_widget.layout().addWidget(query_selection_area)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        central_widget.layout().addWidget(button_box)

        # Set the central widget of the dialog.
        self.setLayout(central_widget.layout())

    def browse_button_clicked(self):
        # This function will be called when the button is clicked.
        # You can implement the logic to select a document or query here.
        (filename, _type) = QFileDialog.getOpenFileName(self, caption="Select PDF Document", directory=".", filter="PDF Files (*.pdf);;All Files (*)")
        # Update the QLineEdit with the selected file path
        self.pdf_selection.setText(filename)

        pass

def get_relevancy_color_tags_from_percentage(percentage):
    modded_cyan = QColor("cyan")
    modded_cyan.setAlphaF(exponential_zero_to_one(percentage))
    black = QColor("black")
    return (modded_cyan, black)

def exponential_zero_to_one(x):
    return (pow(20, x) - 1)/19

# You need one (and only one) QApplication instance per application.
# Pass in sys.argv to allow command line arguments for your app.
# If you know you won't use command line arguments QApplication([]) works too.
app = QApplication(sys.argv)

# Create a Qt widget, which will be our window.
window = MainWindow()
window.show()  # IMPORTANT!!!!! Windows are hidden by default.

# Start the event loop.
app.exec()


# Your application won't reach here until you exit and the event
# loop has stopped.
