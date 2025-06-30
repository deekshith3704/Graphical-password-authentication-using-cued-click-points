import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, 
                            QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit, 
                            QMessageBox, QStackedWidget, QProgressDialog, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QMouseEvent
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
import numpy as np
from database import Database
from deep_learning_models import compare_models, ImageDataset, DeepResidualNetwork, VisionTransformer
import matplotlib.pyplot as plt
import torch
import secrets
import string

class PreTrainingThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path

    def run(self):
        try:
            self.progress.emit("Starting model pre-training...")
            results = compare_models(self.dataset_path, epochs=5, num_images=3)
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

class ImageClickLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.click_points = []
        self.image_path = None
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.click_points.append((pos.x(), pos.y()))
            self._draw_point(pos.x(), pos.y())

    def _draw_point(self, x: int, y: int):
        pixmap = self.pixmap()
        if pixmap:
            image = pixmap.toImage()
            for i in range(-7, 8):
                for j in range(-7, 8):
                    if 0 <= x+i < image.width() and 0 <= y+j < image.height():
                        image.setPixel(x+i, y+j, 0xFF0000FF)
            self.setPixmap(QPixmap.fromImage(image))

    def load_image(self, image_path):
        self.image_path = image_path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        image = cv2.addWeighted(image, 0.7, edges, 0.3, 0)
        
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(600, 400, Qt.KeepAspectRatio)
        self.setPixmap(scaled_pixmap)
        self.click_points = []

class GraphicalPasswordApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = Database()
        self.dataset_path = "dataset"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check if models are already trained
        if self.models_exist():
            self.load_existing_models()
        else:
            # Start pre-training only if models don't exist
            self.pre_train_models()

    def models_exist(self):
        return (os.path.exists('best_DeepResidualNetwork.pth') and 
                os.path.exists('best_VisionTransformer.pth') and
                os.path.exists('training_metrics.png'))

    def load_existing_models(self):
        try:
            # Load the best models
            self.resnet_model = DeepResidualNetwork().to(self.device)
            self.resnet_model.load_state_dict(torch.load('best_DeepResidualNetwork.pth'))
            self.resnet_model.eval()

            self.vit_model = VisionTransformer().to(self.device)
            self.vit_model.load_state_dict(torch.load('best_VisionTransformer.pth'))
            self.vit_model.eval()

            # Initialize the UI after models are loaded
            self.init_ui()
            self.show()

            # Show message that models were loaded
            QMessageBox.information(self, "Models Loaded", 
                                  "Pre-trained models loaded successfully!\n"
                                  "You can view the training metrics in 'training_metrics.png'")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load existing models: {str(e)}")
            # If loading fails, start pre-training
            self.pre_train_models()

    def pre_train_models(self):
        # Create and show progress dialog
        self.progress_dialog = QProgressDialog("Pre-training models...", None, 0, 0, self)
        self.progress_dialog.setWindowTitle("Pre-training")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setAutoClose(True)
        self.progress_dialog.show()

        # Start pre-training in background thread
        self.pre_training_thread = PreTrainingThread(self.dataset_path)
        self.pre_training_thread.progress.connect(self.update_progress)
        self.pre_training_thread.finished.connect(self.on_pre_training_complete)
        self.pre_training_thread.error.connect(self.on_pre_training_error)
        self.pre_training_thread.start()

    def update_progress(self, message):
        self.progress_dialog.setLabelText(message)

    def on_pre_training_complete(self, results):
        """Handle completion of pre-training."""
        self.progress_dialog.close()
        try:
            # Load the best models
            self.resnet_model = DeepResidualNetwork().to(self.device)
            self.resnet_model.load_state_dict(torch.load('best_DeepResidualNetwork.pth'))
            self.resnet_model.eval()

            self.vit_model = VisionTransformer().to(self.device)
            self.vit_model.load_state_dict(torch.load('best_VisionTransformer.pth'))
            self.vit_model.eval()

            # Initialize the UI after models are loaded
            self.init_ui()
            self.show()

            # Show training results with graphs
            self.show_training_results(results)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load trained models: {str(e)}")
            sys.exit(1)

    def show_training_results(self, results):
        """Display the training results."""
        try:
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 12))
            
            # Plot 1: IRR Line Plot (top left)
            ax1 = plt.subplot(2, 2, 1)
            distortion_params = np.linspace(0.0, 0.8, 5)
            
            # Plot IRR lines with distinct colors and styles
            ax1.plot(distortion_params, results['drn_irr_trend'], 
                    'b-', label='ResNet', linewidth=2, color='skyblue')
            ax1.plot(distortion_params, results['vit_irr_trend'], 
                    'g-', label='ViT', linewidth=2, color='lightgreen')
            ax1.plot(distortion_params, results['hybrid_irr_trend'], 
                    'r-', label='Hybrid', linewidth=2, color='salmon')
            ax1.plot(distortion_params, results['existing_gpass_irr_trend'], 
                    'k--', label='Traditional GPass', linewidth=2, color='gray')
            
            ax1.set_xlabel('Distortion parameter')
            ax1.set_ylabel('IRR')
            ax1.set_title('(a) Information Retention Rate (IRR)')
            ax1.grid(True, which='both', linestyle='-', alpha=0.2)
            ax1.legend(fontsize=10, loc='upper right')
            ax1.set_ylim(0.2, 0.9)
            ax1.set_xlim(0.0, 0.8)

            # Plot 2: PDS Line Plot (top right)
            ax2 = plt.subplot(2, 2, 2)
            ax2.plot(distortion_params, results['drn_pds_trend'], 
                    'b-', label='ResNet', linewidth=2, color='skyblue')
            ax2.plot(distortion_params, results['vit_pds_trend'], 
                    'g-', label='ViT', linewidth=2, color='lightgreen')
            ax2.plot(distortion_params, results['hybrid_pds_trend'], 
                    'r-', label='Hybrid', linewidth=2, color='salmon')
            ax2.plot(distortion_params, results['existing_gpass_pds_trend'], 
                    'k--', label='Traditional GPass', linewidth=2, color='gray')
            
            ax2.set_xlabel('Distortion parameter')
            ax2.set_ylabel('PDS')
            ax2.set_title('(b) Password Diversity Score (PDS)')
            ax2.grid(True, which='both', linestyle='-', alpha=0.2)
            ax2.legend(fontsize=10, loc='upper right')
            ax2.set_ylim(0.2, 0.9)
            ax2.set_xlim(0.0, 0.8)

            # Plot 3: IRR Standard Deviation (bottom left)
            ax3 = plt.subplot(2, 2, 3)
            systems = ['ResNet', 'ViT', 'Hybrid', 'Traditional']
            irr_stds = [
                results['drn_irr_std'],
                results['vit_irr_std'],
                results['hybrid_irr_std'],
                results['existing_gpass_irr_std']
            ]
            
            colors = ['skyblue', 'lightgreen', 'salmon', 'gray']
            bars = ax3.bar(systems, irr_stds, color=colors, width=0.6)
            ax3.set_ylabel('Standard deviation')
            ax3.set_title('(c) Standard deviation for Information Retention Rate (IRR)')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, linestyle='-', alpha=0.2, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            ax3.set_ylim(0, 0.18)

            # Plot 4: PDS Standard Deviation (bottom right)
            ax4 = plt.subplot(2, 2, 4)
            pds_stds = [
                results['drn_pds_std'],
                results['vit_pds_std'],
                results['hybrid_pds_std'],
                results['existing_gpass_pds_std']
            ]
            
            bars = ax4.bar(systems, pds_stds, color=colors, width=0.6)
            ax4.set_ylabel('Standard deviation')
            ax4.set_title('(d) Standard deviation for Password Diversity Score (PDS)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, linestyle='-', alpha=0.2, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom')
            
            ax4.set_ylim(0, 0.10)

            # Adjust layout with more space
            plt.tight_layout(pad=2.0)
            
            # Save with high resolution
            plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
            
            # Show the plot immediately
            plt.show()

            # Create detailed performance report
            report = (
                f"Model Performance Report\n\n"
                f"Information Retention Rate (IRR) Analysis:\n"
                f"- ResNet IRR Standard Deviation: {results['drn_irr_std']:.4f}\n"
                f"- ViT IRR Standard Deviation: {results['vit_irr_std']:.4f}\n"
                f"- Hybrid IRR Standard Deviation: {results['hybrid_irr_std']:.4f}\n"
                f"- Traditional GPass IRR Standard Deviation: {results['existing_gpass_irr_std']:.4f}\n\n"
                f"Password Diversity Score (PDS) Analysis:\n"
                f"- ResNet PDS Standard Deviation: {results['drn_pds_std']:.4f}\n"
                f"- ViT PDS Standard Deviation: {results['vit_pds_std']:.4f}\n"
                f"- Hybrid PDS Standard Deviation: {results['hybrid_pds_std']:.4f}\n"
                f"- Traditional GPass PDS Standard Deviation: {results['existing_gpass_pds_std']:.4f}\n\n"
                f"Model Comparison:\n"
                f"- Best IRR Stability: {min(['ResNet', 'ViT', 'Hybrid', 'Traditional'], key=lambda x: results[x.lower().replace('traditional', 'existing_gpass') + '_irr_std'])}\n"
                f"- Best PDS Consistency: {min(['ResNet', 'ViT', 'Hybrid', 'Traditional'], key=lambda x: results[x.lower().replace('traditional', 'existing_gpass') + '_pds_std'])}\n\n"
                f"Training metrics visualization has been saved to 'training_metrics.png'"
            )

            # Show results message with detailed report
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Model Performance Report")
            msg.setText("Models trained successfully!")
            msg.setDetailedText(report)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Could not display training metrics: {str(e)}")
            # Still show the basic results even if plotting fails
            QMessageBox.information(self, "Pre-training Complete", 
                                  f"Models trained successfully!\n\n"
                                  f"ResNet IRR Std: {results['drn_irr_std']:.4f}, PDS Std: {results['drn_pds_std']:.4f}\n"
                                  f"ViT IRR Std: {results['vit_irr_std']:.4f}, PDS Std: {results['vit_pds_std']:.4f}\n"
                                  f"Hybrid IRR Std: {results['hybrid_irr_std']:.4f}, PDS Std: {results['hybrid_pds_std']:.4f}\n"
                                  f"Traditional GPass IRR Std: {results['existing_gpass_irr_std']:.4f}, PDS Std: {results['existing_gpass_pds_std']:.4f}")

    def on_pre_training_error(self, error_message):
        self.progress_dialog.close()
        QMessageBox.critical(self, "Pre-training Error", f"Error during pre-training: {error_message}")
        sys.exit(1)
        
    def init_ui(self):
        self.setWindowTitle('Graphical Password Authentication')
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)
        
        self.create_welcome_screen()
        self.create_auth_screen()
        
        self.stacked_widget.setCurrentIndex(0)
        
    def create_welcome_screen(self):
        welcome_widget = QWidget()
        layout = QVBoxLayout(welcome_widget)
        
        welcome_label = QLabel('Welcome to Graphical Password Authentication')
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText('Enter Username')
        layout.addWidget(self.username_input)
        
        button_layout = QHBoxLayout()
        register_btn = QPushButton('Register')
        login_btn = QPushButton('Login')
        
        register_btn.clicked.connect(lambda: self.start_auth('register'))
        login_btn.clicked.connect(lambda: self.start_auth('login'))
        
        button_layout.addWidget(register_btn)
        button_layout.addWidget(login_btn)
        layout.addLayout(button_layout)
        
        self.stacked_widget.addWidget(welcome_widget)
        
    def create_auth_screen(self):
        auth_widget = QWidget()
        layout = QVBoxLayout(auth_widget)
        
        self.image_label = ImageClickLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        self.instruction_label = QLabel()
        self.instruction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.instruction_label)
        
        self.next_button = QPushButton('Next')
        self.next_button.clicked.connect(self.handle_next)
        layout.addWidget(self.next_button)
        
        self.stacked_widget.addWidget(auth_widget)
        
    def start_auth(self, mode):
        username = self.username_input.text()
        if not username:
            QMessageBox.warning(self, 'Error', 'Please enter a username')
            return
            
        self.username = username
        self.mode = mode
        self.current_image_index = 0
        self.all_click_points = []

        if self.mode == 'register':
            # For registration, get random images
            self.dataset = ImageDataset(self.dataset_path, num_images=3)
            self.image_sequence = self.dataset.get_random_images()
        else:
            # For login, retrieve the stored image sequence from the database
            try:
                stored_data = self.db.get_user_data(self.username)
                if not stored_data:
                    QMessageBox.warning(self, 'Error', 'User not found!')
                    return
                    
                # Get the stored image sequence - it's already JSON decoded in get_user_data
                self.image_sequence = stored_data['image_sequence']
                if not isinstance(self.image_sequence, list):
                    QMessageBox.warning(self, 'Error', 'Invalid image sequence format!')
                    return
                    
                if not self.image_sequence:
                    QMessageBox.warning(self, 'Error', 'No image sequence found for this user!')
                    return
                    
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error retrieving user data: {str(e)}')
                return
        
        if not self.image_sequence:
            QMessageBox.critical(self, 'Error', 'No images available')
            return
            
        self.show_next_image()
        self.stacked_widget.setCurrentIndex(1)
        
    def show_next_image(self):
        if self.current_image_index < len(self.image_sequence):
            image_path = os.path.join(self.dataset_path, self.image_sequence[self.current_image_index])
            self.image_label.load_image(image_path)
            
            action = "registration" if self.mode == 'register' else "login"
            self.instruction_label.setText(
                f'Image {self.current_image_index + 1}/3\n'
                f'Click your points for {action}'
            )
            
    def handle_next(self):
        if not self.image_label.click_points:
            QMessageBox.warning(self, 'Error', 'Please select at least one point')
            return
            
        self.all_click_points.append(self.image_label.click_points)
        self.current_image_index += 1
        
        if self.current_image_index < len(self.image_sequence):
            self.show_next_image()
        else:
            self.complete_auth()
            
    def complete_auth(self):
        if self.mode == 'register':
            # Get security question and answer
            security_question, ok = QInputDialog.getText(
                self, 'Security Question', 
                'Enter your security question (e.g., "What is your mother\'s maiden name?"):'
            )
            if not ok or not security_question:
                QMessageBox.warning(self, 'Error', 'Security question is required!')
                return

            security_answer, ok = QInputDialog.getText(
                self, 'Security Answer', 
                'Enter your answer to the security question:'
            )
            if not ok or not security_answer:
                QMessageBox.warning(self, 'Error', 'Security answer is required!')
                return
            
            success = self.db.register_user(
                self.username,
                security_question,
                security_answer,
                self.all_click_points,
                self.image_sequence
            )
            if success:
                # Show success message
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Information)
                msg.setWindowTitle("Registration Successful")
                msg.setText("Registration successful!")
                msg.setInformativeText(
                    f"Please remember your security question:\n{security_question}\n\n"
                    "You will need this and your click points to login."
                )
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                
                # Return to welcome screen
                self.stacked_widget.setCurrentIndex(0)
                self.username_input.clear()
            else:
                QMessageBox.warning(self, 'Error', 'Registration failed! Username may already exist.')
                self.stacked_widget.setCurrentIndex(0)
        else:
            # For login, get the security question from database
            try:
                user_data = self.db.get_user_data(self.username)
                if not user_data:
                    QMessageBox.warning(self, 'Error', 'User not found!')
                    self.stacked_widget.setCurrentIndex(0)
                    return

                # Show security question and get answer
                security_answer, ok = QInputDialog.getText(
                    self, 'Security Question', 
                    f'Please answer your security question:\n{user_data["security_question"]}'
                )
                if not ok:
                    self.stacked_widget.setCurrentIndex(0)
                    return
                
                success = self.db.authenticate_user(
                    self.username,
                    security_answer,
                    self.all_click_points,
                    self.image_sequence
                )
                if success:
                    # Show success message
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Information)
                    msg.setWindowTitle("Login Successful")
                    msg.setText("Login successful!")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec_()
                    
                    # Return to welcome screen
                    self.stacked_widget.setCurrentIndex(0)
                    self.username_input.clear()
                else:
                    QMessageBox.warning(self, 'Error', 'Authentication failed! Please check your security answer and click points.')
                    self.stacked_widget.setCurrentIndex(0)
            except Exception as e:
                QMessageBox.critical(self, 'Error', f'Error during authentication: {str(e)}')
                self.stacked_widget.setCurrentIndex(0)

def main():
    app = QApplication(sys.argv)
    window = GraphicalPasswordApp()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
