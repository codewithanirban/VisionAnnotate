import sys
import os
import json
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image
import PIL.ImageQt  # Import this way for compatibility

# Set environment variables to avoid OpenGL issues on Jetson
os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["QT_QUICK_BACKEND"] = "software"
os.environ["QT_QPA_PLATFORM"] = "xcb"  # Use XCB platform

class YOLOLabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.image_dir = ""
        self.image_files = []
        self.current_index = -1
        self.current_image = None
        self.labels = []
        self.classes = []
        self.drawing = False
        self.start_point = QPoint()
        self.current_rect = None
        self.selected_label = -1
        self.drawing_mode = True  # Fixed: added missing attribute
        self.edit_mode = False
        self.orientation_mode = False
        self.rotation_angle = 0
        self.scale_factor = 1.0
        self.dragging = False  # Fixed: added missing attribute
        self.rotating = False  # Fixed: added missing attribute
        self.drag_start = None  # Fixed: added missing attribute
        self.orientation_start = None  # Fixed: added missing attribute
        self.resizing = False
        self.resize_handle = -1
        self.drag_offset = QPointF()
        
        # Store original image dimensions
        self.image_width = 0
        self.image_height = 0
        self.original_image = None
        
        # Progress tracking
        self.progress_file = Path.home() / ".yolo_label_tool_progress.json"
        self.labeled_images = set()
        
        self.init_ui()
        self.load_config()
        
    def init_ui(self):
        self.setWindowTitle("YOLO Labeling Tool - Jetson Compatible")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Canvas for image display - use QGraphicsView for better performance
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.Antialiasing)
        self.graphics_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.graphics_view.setDragMode(QGraphicsView.NoDrag)
        
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)
        self.graphics_view.setStyleSheet("background-color: #2b2b2b;")
        
        left_layout.addWidget(self.graphics_view)
        
        # Navigation buttons with progress info
        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        
        self.prev_btn = QPushButton("◀ Previous")
        self.prev_btn.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.next_btn)
        
        self.save_btn = QPushButton("💾 Save Labels")
        self.save_btn.clicked.connect(self.save_labels)
        nav_layout.addWidget(self.save_btn)
        
        # Add skip button for unlabeled images
        self.skip_btn = QPushButton("⏭️ Skip to Next Unlabeled")
        self.skip_btn.clicked.connect(self.skip_to_next_unlabeled)
        nav_layout.addWidget(self.skip_btn)
        
        left_layout.addWidget(nav_widget)
        
        # Right panel for controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setAlignment(Qt.AlignTop)
        
        # Image info with progress
        info_group = QGroupBox("Image Info")
        info_layout = QFormLayout()
        
        self.image_name_label = QLabel("No image loaded")
        info_layout.addRow("Image:", self.image_name_label)
        
        self.image_size_label = QLabel("0 x 0")
        info_layout.addRow("Size:", self.image_size_label)
        
        self.label_count_label = QLabel("0 labels")
        info_layout.addRow("Labels:", self.label_count_label)
        
        self.progress_label = QLabel("Progress: 0/0 (0%)")
        info_layout.addRow("Progress:", self.progress_label)
        
        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)
        
        # Class management
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout()
        
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(150)
        class_layout.addWidget(self.class_list)
        
        class_btn_layout = QHBoxLayout()
        add_class_btn = QPushButton("➕ Add")
        add_class_btn.clicked.connect(self.add_class)
        class_btn_layout.addWidget(add_class_btn)
        
        remove_class_btn = QPushButton("➖ Remove")
        remove_class_btn.clicked.connect(self.remove_class)
        class_btn_layout.addWidget(remove_class_btn)
        
        class_layout.addLayout(class_btn_layout)
        class_group.setLayout(class_layout)
        right_layout.addWidget(class_group)
        
        # Labels list
        labels_group = QGroupBox("Labels")
        labels_layout = QVBoxLayout()
        
        self.labels_list = QListWidget()
        self.labels_list.setMaximumHeight(200)
        self.labels_list.itemClicked.connect(self.select_label)
        labels_layout.addWidget(self.labels_list)
        
        labels_btn_layout = QHBoxLayout()
        delete_label_btn = QPushButton("🗑️ Delete")
        delete_label_btn.clicked.connect(self.delete_label)
        labels_btn_layout.addWidget(delete_label_btn)
        
        clear_labels_btn = QPushButton("🗑️ Clear All")
        clear_labels_btn.clicked.connect(self.clear_all_labels)
        labels_btn_layout.addWidget(clear_labels_btn)
        
        labels_layout.addLayout(labels_btn_layout)
        labels_group.setLayout(labels_layout)
        right_layout.addWidget(labels_group)
        
        # Mode selection
        mode_group = QGroupBox("Mode")
        mode_layout = QHBoxLayout()
        
        self.draw_mode_btn = QPushButton("✏️ Draw")
        self.draw_mode_btn.setCheckable(True)
        self.draw_mode_btn.setChecked(True)
        self.draw_mode_btn.clicked.connect(self.set_draw_mode)
        mode_layout.addWidget(self.draw_mode_btn)
        
        self.edit_mode_btn = QPushButton("✏️ Edit")
        self.edit_mode_btn.setCheckable(True)
        self.edit_mode_btn.clicked.connect(self.set_edit_mode)
        mode_layout.addWidget(self.edit_mode_btn)
        
        self.orientation_mode_btn = QPushButton("🔄 Rotate")
        self.orientation_mode_btn.setCheckable(True)
        self.orientation_mode_btn.clicked.connect(self.set_orientation_mode)
        mode_layout.addWidget(self.orientation_mode_btn)
        
        mode_group.setLayout(mode_layout)
        right_layout.addWidget(mode_group)
        
        # Zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()
        
        zoom_out_btn = QPushButton("➖")
        zoom_out_btn.clicked.connect(self.zoom_out)
        zoom_layout.addWidget(zoom_out_btn)
        
        zoom_reset_btn = QPushButton("Reset")
        zoom_reset_btn.clicked.connect(self.zoom_reset)
        zoom_layout.addWidget(zoom_reset_btn)
        
        zoom_in_btn = QPushButton("➕")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_layout.addWidget(zoom_in_btn)
        
        zoom_group.setLayout(zoom_layout)
        right_layout.addWidget(zoom_group)
        
        # Label info
        label_info_group = QGroupBox("Label Info")
        label_info_layout = QFormLayout()
        
        self.class_combo = QComboBox()
        label_info_layout.addRow("Class:", self.class_combo)
        
        coords_layout = QHBoxLayout()
        self.x_edit = QLineEdit()
        self.x_edit.setPlaceholderText("x")
        self.x_edit.setMaximumWidth(80)
        coords_layout.addWidget(self.x_edit)
        
        self.y_edit = QLineEdit()
        self.y_edit.setPlaceholderText("y")
        self.y_edit.setMaximumWidth(80)
        coords_layout.addWidget(self.y_edit)
        label_info_layout.addRow("Center:", coords_layout)
        
        size_layout = QHBoxLayout()
        self.width_edit = QLineEdit()
        self.width_edit.setPlaceholderText("w")
        self.width_edit.setMaximumWidth(80)
        size_layout.addWidget(self.width_edit)
        
        self.height_edit = QLineEdit()
        self.height_edit.setPlaceholderText("h")
        self.height_edit.setMaximumWidth(80)
        size_layout.addWidget(self.height_edit)
        label_info_layout.addRow("Size:", size_layout)
        
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(-180, 180)
        self.angle_slider.setValue(0)
        self.angle_slider.valueChanged.connect(self.update_angle_from_slider)
        label_info_layout.addRow("Angle:", self.angle_slider)
        
        self.angle_label = QLabel("0°")
        label_info_layout.addRow("", self.angle_label)
        
        update_btn = QPushButton("📝 Update Label")
        update_btn.clicked.connect(self.update_label)
        label_info_layout.addRow(update_btn)
        
        label_info_group.setLayout(label_info_layout)
        right_layout.addWidget(label_info_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 3)
        main_layout.addWidget(right_panel, 1)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("📁 File")
        
        open_dir_action = QAction("📂 Open Image Directory", self)
        open_dir_action.triggered.connect(self.open_image_dir)
        open_dir_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_dir_action)
        
        load_classes_action = QAction("📖 Load Classes", self)
        load_classes_action.triggered.connect(self.load_classes_file)
        file_menu.addAction(load_classes_action)
        
        save_classes_action = QAction("💾 Save Classes", self)
        save_classes_action.triggered.connect(self.save_classes_file)
        file_menu.addAction(save_classes_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("📤 Export All Labels", self)
        export_action.triggered.connect(self.export_all_labels)
        file_menu.addAction(export_action)
        
        exit_action = QAction("🚪 Exit", self)
        exit_action.triggered.connect(self.close)
        exit_action.setShortcut("Ctrl+Q")
        file_menu.addAction(exit_action)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Mouse tracking
        self.graphics_view.setMouseTracking(True)
        self.graphics_view.viewport().installEventFilter(self)
        
        # Temporary rectangle for drawing
        self.temp_rect = None
        
        # Edit mode handles
        self.handles = []
        self.selected_handle = -1
        
    def eventFilter(self, source, event):
        """Handle mouse events on the graphics view"""
        if source is self.graphics_view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                return self.handle_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self.handle_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self.handle_mouse_release(event)
            elif event.type() == QEvent.Wheel:
                return self.handle_wheel(event)
        return super().eventFilter(source, event)
    
    def handle_mouse_press(self, event):
        """Handle mouse press on canvas"""
        if event.button() == Qt.LeftButton and self.current_image:
            pos = self.graphics_view.mapToScene(event.pos())
            x, y = pos.x(), pos.y()
            
            if self.drawing_mode:
                # Start drawing a new bounding box
                self.drawing = True
                self.start_point = QPointF(x, y)
                self.temp_rect = QRectF(x, y, 0, 0)
                return True
            
            elif self.edit_mode and self.selected_label >= 0:
                # Check if clicking on resize handles
                label = self.labels[self.selected_label]
                handles = self.calculate_handles(label)
                
                for i, (hx, hy) in enumerate(handles):
                    if abs(x - hx) < 10 and abs(y - hy) < 10:
                        self.resizing = True
                        self.resize_handle = i
                        self.drag_start = QPointF(x, y)
                        self.original_label = label.copy()
                        return True
                
                # Check if clicking inside the bounding box to drag it
                x_center = label['x_center'] * self.image_width
                y_center = label['y_center'] * self.image_height
                width = label['width'] * self.image_width
                height = label['height'] * self.image_height
                
                rect = QRectF(x_center - width/2, y_center - height/2, width, height)
                if rect.contains(x, y):
                    self.dragging = True
                    self.drag_start = QPointF(x, y)
                    self.drag_offset = QPointF(x - x_center, y - y_center)
                    self.original_label = label.copy()
                    return True
                
                # Check if clicking on other labels to select them
                for i, label in enumerate(self.labels):
                    x_center = label['x_center'] * self.image_width
                    y_center = label['y_center'] * self.image_height
                    width = label['width'] * self.image_width
                    height = label['height'] * self.image_height
                    
                    rect = QRectF(x_center - width/2, y_center - height/2, width, height)
                    if rect.contains(x, y):
                        self.selected_label = i
                        self.update_labels_list()
                        self.update_label_info()
                        self.update_display()
                        return True
            
            elif self.orientation_mode and self.selected_label >= 0:
                # Start rotating selected label
                self.rotating = True
                self.rotation_start = pos
                return True
        
        return False
    
    def calculate_handles(self, label):
        """Calculate positions of resize handles for a label"""
        x_center = label['x_center'] * self.image_width
        y_center = label['y_center'] * self.image_height
        width = label['width'] * self.image_width
        height = label['height'] * self.image_height
        angle_rad = np.radians(label['angle'])
        
        # Corner points relative to center
        corners_rel = [
            (-width/2, -height/2),  # Top-left
            (width/2, -height/2),   # Top-right
            (width/2, height/2),    # Bottom-right
            (-width/2, height/2)    # Bottom-left
        ]
        
        # Rotate corners
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        handles = []
        
        for dx, dy in corners_rel:
            x_rot = dx * cos_a - dy * sin_a + x_center
            y_rot = dx * sin_a + dy * cos_a + y_center
            handles.append((x_rot, y_rot))
        
        # Edge centers
        edge_centers = [
            (0, -height/2),  # Top
            (width/2, 0),    # Right
            (0, height/2),   # Bottom
            (-width/2, 0)    # Left
        ]
        
        for dx, dy in edge_centers:
            x_rot = dx * cos_a - dy * sin_a + x_center
            y_rot = dx * sin_a + dy * cos_a + y_center
            handles.append((x_rot, y_rot))
        
        return handles
    
    def handle_mouse_move(self, event):
        """Handle mouse movement on canvas"""
        pos = self.graphics_view.mapToScene(event.pos())
        x, y = pos.x(), pos.y()
        
        if self.drawing and self.drawing_mode:
            self.temp_rect = QRectF(self.start_point, pos).normalized()
            self.update_display_with_temp()
            return True
        
        elif self.dragging and self.edit_mode and self.selected_label >= 0:
            # Drag the entire bounding box
            label = self.labels[self.selected_label]
            original = self.original_label
            
            # Calculate new center position
            new_x_center = (x - self.drag_offset.x()) / self.image_width
            new_y_center = (y - self.drag_offset.y()) / self.image_height
            
            # Keep within bounds
            new_x_center = max(0.0, min(1.0, new_x_center))
            new_y_center = max(0.0, min(1.0, new_y_center))
            
            label['x_center'] = new_x_center
            label['y_center'] = new_y_center
            
            self.update_display()
            self.update_label_info()
            self.update_labels_list()
            return True
        
        elif self.resizing and self.edit_mode and self.selected_label >= 0:
            # Resize the bounding box from a specific handle
            label = self.labels[self.selected_label]
            original = self.original_label
            
            # Get original corners
            orig_corners = self.calculate_handles(original)[:4]
            
            # Create modified corner based on handle being dragged
            modified_corners = list(orig_corners)
            modified_corners[self.resize_handle] = (x, y)
            
            # Calculate new bounding box from modified corners
            if self.resize_handle < 4:  # Corner handle
                # For simplicity, calculate axis-aligned bounding box first
                # This could be improved for rotated boxes
                xs = [p[0] for p in modified_corners]
                ys = [p[1] for p in modified_corners]
                
                new_x_min = min(xs) / self.image_width
                new_y_min = min(ys) / self.image_height
                new_x_max = max(xs) / self.image_width
                new_y_max = max(ys) / self.image_height
                
                label['x_center'] = (new_x_min + new_x_max) / 2
                label['y_center'] = (new_y_min + new_y_max) / 2
                label['width'] = new_x_max - new_x_min
                label['height'] = new_y_max - new_y_min
            else:  # Edge handle (handles 4-7)
                edge_idx = self.resize_handle - 4
                # Adjust width or height based on which edge
                if edge_idx == 0 or edge_idx == 2:  # Top or bottom edge
                    scale = (y - (original['y_center'] * self.image_height)) / (self.image_height * original['height'] / 2)
                    label['height'] = original['height'] * (1 + scale)
                else:  # Left or right edge
                    scale = (x - (original['x_center'] * self.image_width)) / (self.image_width * original['width'] / 2)
                    label['width'] = original['width'] * (1 + scale)
            
            # Keep within bounds
            label['x_center'] = max(0.0, min(1.0, label['x_center']))
            label['y_center'] = max(0.0, min(1.0, label['y_center']))
            label['width'] = max(0.001, min(1.0, label['width']))
            label['height'] = max(0.001, min(1.0, label['height']))
            
            self.update_display()
            self.update_label_info()
            self.update_labels_list()
            return True
        
        elif self.rotating and self.orientation_mode and self.selected_label >= 0:
            # Rotate selected label
            label = self.labels[self.selected_label]
            center_x = label['x_center'] * self.image_width
            center_y = label['y_center'] * self.image_height
            
            # Calculate angle from center to mouse position
            dx = x - center_x
            dy = y - center_y
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Update label
            label['angle'] = angle
            self.angle_slider.setValue(int(angle))
            self.angle_label.setText(f"{angle:.1f}°")
            self.update_display()
            self.update_labels_list()
            return True
        
        # Update cursor based on hover position
        if self.edit_mode and self.current_image:
            # Check if hovering over handles
            for i, label in enumerate(self.labels):
                handles = self.calculate_handles(label)
                for hx, hy in handles:
                    if abs(x - hx) < 10 and abs(y - hy) < 10:
                        self.graphics_view.viewport().setCursor(Qt.SizeAllCursor)
                        return True
            
            # Check if hovering inside a label
            for label in self.labels:
                x_center = label['x_center'] * self.image_width
                y_center = label['y_center'] * self.image_height
                width = label['width'] * self.image_width
                height = label['height'] * self.image_height
                
                rect = QRectF(x_center - width/2, y_center - height/2, width, height)
                if rect.contains(x, y):
                    self.graphics_view.viewport().setCursor(Qt.OpenHandCursor)
                    return True
            
            self.graphics_view.viewport().setCursor(Qt.ArrowCursor)
        
        return False
    
    def handle_mouse_release(self, event):
        """Handle mouse release on canvas"""
        if event.button() == Qt.LeftButton:
            if self.drawing and self.temp_rect and self.drawing_mode:
                # Finalize the bounding box
                rect = self.temp_rect
                
                # Calculate normalized YOLO coordinates
                x_center = (rect.x() + rect.width() / 2) / self.image_width
                y_center = (rect.y() + rect.height() / 2) / self.image_height
                width = rect.width() / self.image_width
                height = rect.height() / self.image_height
                
                # Get class from combo box
                class_id = self.class_combo.currentIndex()
                if class_id < 0:
                    class_id = 0
                
                # Add new label
                self.labels.append({
                    'class_id': class_id,
                    'x_center': max(0.0, min(1.0, x_center)),
                    'y_center': max(0.0, min(1.0, y_center)),
                    'width': max(0.001, min(1.0, width)),
                    'height': max(0.001, min(1.0, height)),
                    'angle': self.angle_slider.value()
                })
                
                self.drawing = False
                self.temp_rect = None
                self.update_labels_list()
                self.update_display()
                self.status_bar.showMessage(f"Added label for {self.classes[class_id] if class_id < len(self.classes) else 'Unknown'}", 2000)
                return True
            
            self.dragging = False
            self.resizing = False
            self.rotating = False
            self.resize_handle = -1
            
            # Update progress when label is modified
            if self.current_index >= 0:
                self.update_progress()
        
        return False
    
    def handle_wheel(self, event):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.graphics_view.scale(1.1, 1.1)
        else:
            self.graphics_view.scale(0.9, 0.9)
        return True
    
    def load_config(self):
        """Load configuration and classes"""
        config_path = Path.home() / ".yolo_label_tool_jetson.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.classes = config.get('classes', [])
                    if not self.classes:
                        self.classes = ["object"]
            except:
                self.classes = ["object"]
        else:
            self.classes = ["object"]
        
        # Load progress
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress_data = json.load(f)
                    self.labeled_images = set(progress_data.get('labeled_images', []))
            except:
                self.labeled_images = set()
        
        self.update_class_list()
    
    def save_config(self):
        """Save configuration and classes"""
        config_path = Path.home() / ".yolo_label_tool_jetson.json"
        config = {'classes': self.classes}
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save progress
        progress_data = {
            'labeled_images': list(self.labeled_images)
        }
        with open(self.progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def open_image_dir(self):
        """Open directory containing images and resume from last unlabeled"""
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            self.image_dir = directory
            self.image_files = []
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG']
            
            for file in sorted(os.listdir(directory)):
                if any(file.lower().endswith(ext) for ext in extensions):
                    self.image_files.append(file)
            
            if self.image_files:
                # Scan for existing labels
                self.scan_existing_labels()
                
                # Find first unlabeled image
                unlabeled_indices = []
                for i, file in enumerate(self.image_files):
                    base_name = os.path.splitext(file)[0]
                    label_file = os.path.join(directory, f"{base_name}.txt")
                    if not os.path.exists(label_file):
                        unlabeled_indices.append(i)
                
                if unlabeled_indices:
                    # Start from first unlabeled image
                    self.current_index = unlabeled_indices[0]
                    self.status_bar.showMessage(f"Resuming from image {self.current_index + 1}/{len(self.image_files)} (unlabeled)")
                else:
                    # All images are labeled, start from first
                    self.current_index = 0
                    self.status_bar.showMessage(f"All {len(self.image_files)} images are already labeled")
                
                self.load_image()
                self.update_progress()
            else:
                QMessageBox.warning(self, "No Images", "No image files found in directory")
    
    def scan_existing_labels(self):
        """Scan directory for existing label files to track progress"""
        self.labeled_images.clear()
        for file in self.image_files:
            base_name = os.path.splitext(file)[0]
            label_file = os.path.join(self.image_dir, f"{base_name}.txt")
            if os.path.exists(label_file):
                self.labeled_images.add(file)
    
    def update_progress(self):
        """Update progress display"""
        if self.image_files:
            total = len(self.image_files)
            labeled = len(self.labeled_images)
            percentage = (labeled / total * 100) if total > 0 else 0
            self.progress_label.setText(f"Progress: {labeled}/{total} ({percentage:.1f}%)")
            
            # Auto-save progress
            self.save_config()
    
    def load_image(self):
        """Load current image"""
        if 0 <= self.current_index < len(self.image_files):
            image_path = os.path.join(self.image_dir, self.image_files[self.current_index])
            
            try:
                # Load image using OpenCV (better compatibility on Jetson)
                cv_image = cv2.imread(image_path)
                if cv_image is None:
                    # Fallback to PIL if OpenCV fails
                    pil_image = Image.open(image_path)
                    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
                if cv_image is None:
                    raise ValueError(f"Failed to load image: {image_path}")
                
                # Convert to QImage
                height, width, channel = cv_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                
                self.current_image = QPixmap.fromImage(q_image)
                self.image_width = width
                self.image_height = height
                self.original_image = self.current_image.copy()
                
                # Clear scene and add image
                self.scene.clear()
                self.scene.addPixmap(self.current_image)
                self.scene.setSceneRect(0, 0, width, height)
                
                # Reset view
                self.graphics_view.resetTransform()
                self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
                
                # Load labels
                self.load_yolo_labels()
                
                # Update UI
                self.update_display()
                self.image_name_label.setText(self.image_files[self.current_index])
                self.image_size_label.setText(f"{width} x {height}")
                self.label_count_label.setText(f"{len(self.labels)} labels")
                
                self.setWindowTitle(f"YOLO Labeling Tool - {self.image_files[self.current_index]} ({self.current_index + 1}/{len(self.image_files)})")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def load_yolo_labels(self):
        """Load YOLO format labels for current image"""
        self.labels = []
        if not self.image_files or self.current_index < 0:
            return
        
        label_path = os.path.join(self.image_dir, 
                                 os.path.splitext(self.image_files[self.current_index])[0] + '.txt')
        
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                angle = float(parts[5]) if len(parts) > 5 else 0.0
                                
                                self.labels.append({
                                    'class_id': class_id,
                                    'x_center': x_center,
                                    'y_center': y_center,
                                    'width': width,
                                    'height': height,
                                    'angle': angle
                                })
                            except ValueError:
                                continue
            except Exception as e:
                self.status_bar.showMessage(f"Error loading labels: {str(e)}", 5000)
        
        self.update_labels_list()
    
    def save_labels(self):
        """Save labels in YOLO format and update progress"""
        if self.current_index < 0 or not self.image_files:
            return
        
        # Save to the same directory as images
        label_path = os.path.join(self.image_dir, 
                                 os.path.splitext(self.image_files[self.current_index])[0] + '.txt')
        
        try:
            with open(label_path, 'w') as f:
                for label in self.labels:
                    f.write(f"{label['class_id']} {label['x_center']:.6f} {label['y_center']:.6f} "
                           f"{label['width']:.6f} {label['height']:.6f} {label['angle']:.6f}\n")
            
            # Update progress tracking
            self.labeled_images.add(self.image_files[self.current_index])
            self.update_progress()
            
            self.status_bar.showMessage(f"Labels saved to {os.path.basename(label_path)}", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save labels: {str(e)}")
    
    def update_display(self):
        """Update the display with image and labels"""
        if not self.current_image:
            return
        
        # Clear and redraw
        self.scene.clear()
        self.scene.addPixmap(self.current_image)
        
        # Draw all labels
        for i, label in enumerate(self.labels):
            # Convert normalized coordinates to pixel coordinates
            x_center = label['x_center'] * self.image_width
            y_center = label['y_center'] * self.image_height
            width = label['width'] * self.image_width
            height = label['height'] * self.image_height
            angle = label['angle']
            
            # Create rectangle
            rect = QRectF(x_center - width/2, y_center - height/2, width, height)
            
            # Create graphics item
            rect_item = QGraphicsRectItem(rect)
            
            # Set color based on selection
            if i == self.selected_label:
                pen_color = QColor(255, 0, 0)  # Red for selected
                pen_width = 3
                brush_color = QColor(255, 0, 0, 30)
                
                # Draw resize handles in edit mode
                if self.edit_mode:
                    handles = self.calculate_handles(label)
                    for hx, hy in handles:
                        handle_item = QGraphicsEllipseItem(hx - 5, hy - 5, 10, 10)
                        handle_item.setPen(QPen(QColor(255, 255, 0), 2))
                        handle_item.setBrush(QBrush(QColor(255, 255, 0, 200)))
                        self.scene.addItem(handle_item)
            else:
                pen_color = QColor(0, 255, 0)  # Green for others
                pen_width = 2
                brush_color = QColor(0, 255, 0, 30)
            
            rect_item.setPen(QPen(pen_color, pen_width))
            rect_item.setBrush(QBrush(brush_color))
            
            # Apply rotation if needed
            if angle != 0:
                rect_item.setTransformOriginPoint(x_center, y_center)
                rect_item.setRotation(angle)
            
            # Add to scene
            self.scene.addItem(rect_item)
            
            # Add class label
            if label['class_id'] < len(self.classes):
                class_name = self.classes[label['class_id']]
            else:
                class_name = f"Class {label['class_id']}"
            
            text_item = QGraphicsTextItem(f"{class_name} ∠{angle:.1f}°")
            text_item.setDefaultTextColor(Qt.white)
            text_item.setFont(QFont("Arial", 10))
            
            # Position text above rectangle
            text_rect = text_item.boundingRect()
            text_item.setPos(rect.x(), rect.y() - text_rect.height())
            
            # Add background to text
            bg_item = QGraphicsRectItem(text_rect)
            bg_item.setPos(text_item.pos())
            bg_item.setBrush(QBrush(QColor(0, 0, 0, 180)))
            bg_item.setPen(QPen(Qt.NoPen))
            
            self.scene.addItem(bg_item)
            self.scene.addItem(text_item)
    
    def update_display_with_temp(self):
        """Update display with temporary drawing rectangle"""
        self.update_display()
        
        if self.temp_rect:
            # Draw temporary rectangle
            rect_item = QGraphicsRectItem(self.temp_rect)
            rect_item.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            rect_item.setBrush(QBrush(QColor(255, 255, 0, 30)))
            self.scene.addItem(rect_item)
    
    def update_labels_list(self):
        """Update the labels list widget"""
        self.labels_list.clear()
        for i, label in enumerate(self.labels):
            if label['class_id'] < len(self.classes):
                class_name = self.classes[label['class_id']]
            else:
                class_name = f"Class {label['class_id']}"
            
            item_text = f"{i}: {class_name} (x:{label['x_center']:.3f}, y:{label['y_center']:.3f})"
            if label['angle'] != 0:
                item_text += f" ∠{label['angle']:.1f}°"
            
            self.labels_list.addItem(item_text)
    
    def select_label(self, item):
        """Select a label from the list"""
        index = self.labels_list.row(item)
        if 0 <= index < len(self.labels):
            self.selected_label = index
            self.update_label_info()
            self.update_display()
    
    def update_label_info(self):
        """Update label info fields with selected label data"""
        if self.selected_label >= 0 and self.selected_label < len(self.labels):
            label = self.labels[self.selected_label]
            
            # Update text fields
            self.x_edit.setText(f"{label['x_center']:.6f}")
            self.y_edit.setText(f"{label['y_center']:.6f}")
            self.width_edit.setText(f"{label['width']:.6f}")
            self.height_edit.setText(f"{label['height']:.6f}")
            
            # Update angle slider
            self.angle_slider.setValue(int(label['angle']))
            self.angle_label.setText(f"{label['angle']:.1f}°")
            
            # Update class combo
            if label['class_id'] < self.class_combo.count():
                self.class_combo.setCurrentIndex(label['class_id'])
    
    def add_class(self):
        """Add a new class"""
        name, ok = QInputDialog.getText(self, "Add Class", "Enter class name:")
        if ok and name:
            self.classes.append(name)
            self.update_class_list()
            self.save_config()
    
    def remove_class(self):
        """Remove selected class"""
        current_row = self.class_list.currentRow()
        if current_row >= 0:
            reply = QMessageBox.question(self, "Remove Class", 
                                        f"Remove class '{self.classes[current_row]}'?\n\n"
                                        "Note: This will not remove labels, but their class IDs may change.",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.classes.pop(current_row)
                self.update_class_list()
                self.save_config()
    
    def update_class_list(self):
        """Update the class list widget"""
        self.class_list.clear()
        self.class_list.addItems(self.classes)
        
        self.class_combo.clear()
        self.class_combo.addItems(self.classes)
    
    def set_draw_mode(self):
        """Set to draw mode"""
        self.drawing_mode = True
        self.edit_mode = False
        self.orientation_mode = False
        
        self.draw_mode_btn.setChecked(True)
        self.edit_mode_btn.setChecked(False)
        self.orientation_mode_btn.setChecked(False)
        
        self.graphics_view.viewport().setCursor(Qt.CrossCursor)
        self.status_bar.showMessage("Draw Mode: Click and drag to draw bounding boxes")
    
    def set_edit_mode(self):
        """Set to edit mode"""
        self.drawing_mode = False
        self.edit_mode = True
        self.orientation_mode = False
        
        self.draw_mode_btn.setChecked(False)
        self.edit_mode_btn.setChecked(True)
        self.orientation_mode_btn.setChecked(False)
        
        self.graphics_view.viewport().setCursor(Qt.ArrowCursor)
        self.status_bar.showMessage("Edit Mode: Drag corners to resize, drag center to move")
    
    def set_orientation_mode(self):
        """Set to orientation mode"""
        self.drawing_mode = False
        self.edit_mode = False
        self.orientation_mode = True
        
        self.draw_mode_btn.setChecked(False)
        self.edit_mode_btn.setChecked(False)
        self.orientation_mode_btn.setChecked(True)
        
        self.graphics_view.viewport().setCursor(Qt.SizeAllCursor)
        self.status_bar.showMessage("Orientation Mode: Drag to rotate selected label")
    
    def update_angle_from_slider(self):
        """Update angle from slider value"""
        angle = self.angle_slider.value()
        self.angle_label.setText(f"{angle}°")
        
        if self.selected_label >= 0:
            self.labels[self.selected_label]['angle'] = angle
            self.update_display()
            self.update_labels_list()
    
    def update_label(self):
        """Update selected label with edited values"""
        if self.selected_label >= 0:
            try:
                # Get values from text fields
                x_center = float(self.x_edit.text())
                y_center = float(self.y_edit.text())
                width = float(self.width_edit.text())
                height = float(self.height_edit.text())
                
                # Validate values
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.001, min(1.0, width))
                height = max(0.001, min(1.0, height))
                
                # Update label
                self.labels[self.selected_label].update({
                    'class_id': self.class_combo.currentIndex(),
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'angle': self.angle_slider.value()
                })
                
                # Update UI
                self.update_display()
                self.update_labels_list()
                self.status_bar.showMessage("Label updated", 2000)
                
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers")
    
    def delete_label(self):
        """Delete selected label"""
        if self.selected_label >= 0:
            self.labels.pop(self.selected_label)
            self.selected_label = -1
            self.update_display()
            self.update_labels_list()
            self.clear_label_info()
            self.status_bar.showMessage("Label deleted", 2000)
    
    def clear_all_labels(self):
        """Clear all labels for current image"""
        if self.labels:
            reply = QMessageBox.question(self, "Clear All Labels", 
                                        "Remove all labels for this image?",
                                        QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.labels.clear()
                self.selected_label = -1
                self.update_display()
                self.update_labels_list()
                self.clear_label_info()
                self.status_bar.showMessage("All labels cleared", 2000)
    
    def clear_label_info(self):
        """Clear label info fields"""
        self.x_edit.clear()
        self.y_edit.clear()
        self.width_edit.clear()
        self.height_edit.clear()
        self.angle_slider.setValue(0)
        self.angle_label.setText("0°")
    
    def prev_image(self):
        """Go to previous image"""
        if self.current_index > 0:
            self.save_labels()
            self.current_index -= 1
            self.load_image()
    
    def next_image(self):
        """Go to next image"""
        if self.current_index < len(self.image_files) - 1:
            self.save_labels()
            self.current_index += 1
            self.load_image()
    
    def skip_to_next_unlabeled(self):
        """Skip to the next image without labels"""
        if not self.image_files:
            return
        
        # Save current labels first
        self.save_labels()
        
        # Find next unlabeled image
        for i in range(self.current_index + 1, len(self.image_files)):
            base_name = os.path.splitext(self.image_files[i])[0]
            label_file = os.path.join(self.image_dir, f"{base_name}.txt")
            if not os.path.exists(label_file):
                self.current_index = i
                self.load_image()
                self.status_bar.showMessage(f"Skipped to unlabeled image {i + 1}/{len(self.image_files)}")
                return
        
        # Check from beginning if none found after current
        for i in range(0, self.current_index):
            base_name = os.path.splitext(self.image_files[i])[0]
            label_file = os.path.join(self.image_dir, f"{base_name}.txt")
            if not os.path.exists(label_file):
                self.current_index = i
                self.load_image()
                self.status_bar.showMessage(f"Skipped to unlabeled image {i + 1}/{len(self.image_files)} (wrapped)")
                return
        
        self.status_bar.showMessage("All images are labeled!", 3000)
    
    def load_classes_file(self):
        """Load classes from file"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Classes", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    self.classes = [line.strip() for line in f if line.strip()]
                self.update_class_list()
                self.save_config()
                self.status_bar.showMessage(f"Loaded {len(self.classes)} classes from {os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load classes: {str(e)}")
    
    def save_classes_file(self):
        """Save classes to file"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Classes", "classes.txt", "Text Files (*.txt);;All Files (*)")
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    for class_name in self.classes:
                        f.write(f"{class_name}\n")
                self.status_bar.showMessage(f"Saved {len(self.classes)} classes to {os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save classes: {str(e)}")
    
    def export_all_labels(self):
        """Export all labels to a directory"""
        if not self.image_dir:
            QMessageBox.warning(self, "No Directory", "Please open an image directory first")
            return
        
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        # Save all labels first
        for i in range(len(self.image_files)):
            self.current_index = i
            self.load_yolo_labels()
            self.save_labels()
        
        # Reset to current image
        self.load_image()
        
        QMessageBox.information(self, "Export Complete", 
                              f"All labels have been saved in the original directory.\n\n"
                              f"You can find them alongside your images.")
    
    def zoom_in(self):
        """Zoom in"""
        self.graphics_view.scale(1.2, 1.2)
    
    def zoom_out(self):
        """Zoom out"""
        self.graphics_view.scale(0.8, 0.8)
    
    def zoom_reset(self):
        """Reset zoom"""
        self.graphics_view.resetTransform()
        self.graphics_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def closeEvent(self, event):
        """Handle application close"""
        reply = QMessageBox.question(self, "Exit", 
                                    "Save labels before exiting?",
                                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        
        if reply == QMessageBox.Yes:
            self.save_labels()
            self.save_config()
            event.accept()
        elif reply == QMessageBox.No:
            event.accept()
        else:
            event.ignore()

def main():
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Labeling Tool")
    
    # Use Fusion style for better look
    app.setStyle("Fusion")
    
    # Create and show window
    window = YOLOLabelTool()
    window.show()
    
    # Start event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()