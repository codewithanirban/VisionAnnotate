import sys
import os
import json
import math
from pathlib import Path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PIL import Image, ImageQt
import numpy as np
from yolo_label_tool import *

class RotatedYOLOLabelTool(YOLOLabelTool):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Rotated Bounding Box Labeling Tool")
        
        # Add orientation visualization
        self.orientation_arrow_length = 50
        self.show_orientation = True
        
        # Add rotation center point for better control
        self.rotation_center = None
        
    def update_display(self):
        if self.current_image:
            display_pixmap = self.current_image.copy()
            painter = QPainter(display_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            
            for i, label in enumerate(self.labels):
                # Convert normalized coordinates to pixel coordinates
                x_center = label['x_center'] * self.image_width
                y_center = label['y_center'] * self.image_height
                width = label['width'] * self.image_width
                height = label['height'] * self.image_height
                angle = label['angle']
                
                # Set colors
                if i == self.selected_label:
                    pen_color = QColor(255, 0, 0)
                    brush_color = QColor(255, 0, 0, 30)
                    pen_width = 3
                else:
                    pen_color = QColor(0, 255, 0)
                    brush_color = QColor(0, 255, 0, 30)
                    pen_width = 2
                
                # Draw rotated rectangle
                painter.save()
                painter.translate(x_center, y_center)
                painter.rotate(angle)
                
                # Draw rectangle
                painter.setPen(QPen(pen_color, pen_width))
                painter.setBrush(QBrush(brush_color))
                painter.drawRect(QRectF(-width/2, -height/2, width, height))
                
                # Draw orientation indicators
                if self.show_orientation:
                    # Main orientation arrow
                    arrow_len = min(width, height) / 3
                    painter.setPen(QPen(QColor(255, 255, 0), 3))
                    painter.drawLine(0, 0, arrow_len, 0)
                    
                    # Arrow head
                    painter.drawLine(arrow_len, 0, arrow_len - 8, -5)
                    painter.drawLine(arrow_len, 0, arrow_len - 8, 5)
                    
                    # Draw rotation center
                    painter.setPen(QPen(QColor(0, 255, 255), 2))
                    painter.drawEllipse(QPointF(0, 0), 3, 3)
                    
                    # Draw axes
                    painter.setPen(QPen(QColor(255, 0, 255, 100), 1))
                    painter.drawLine(-width/2, 0, width/2, 0)  # X-axis
                    painter.drawLine(0, -height/2, 0, height/2)  # Y-axis
                
                painter.restore()
                
                # Draw class label
                if label['class_id'] < len(self.classes):
                    class_name = self.classes[label['class_id']]
                else:
                    class_name = f"Class {label['class_id']}"
                
                font = painter.font()
                font.setPointSize(10)
                painter.setFont(font)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.setBrush(QBrush(QColor(0, 0, 0, 200)))
                
                # Calculate label position (top-left of bounding box after rotation)
                # For simplicity, we'll place it near the bounding box
                label_rect = QRectF(x_center - 50, y_center - height/2 - 25, 100, 25)
                painter.drawRect(label_rect)
                painter.drawText(label_rect, Qt.AlignCenter, f"{class_name} ∠{angle:.1f}°")
            
            painter.end()
            
            # Scale if needed
            if self.scale_factor != 1.0:
                display_pixmap = display_pixmap.scaled(
                    display_pixmap.size() * self.scale_factor,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            
            self.canvas.setPixmap(display_pixmap)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.current_image:
            pos = self.canvas.mapFrom(self, event.pos())
            
            # Adjust for scaling
            if self.scale_factor != 1.0:
                pos = QPoint(int(pos.x() / self.scale_factor), int(pos.y() / self.scale_factor))
            
            if self.drawing_mode:
                # Check if we're near an existing point to adjust rotation center
                self.check_near_existing_points(pos)
                self.drawing = True
                self.start_point = pos
                self.current_rect = QRect(pos, pos)
            elif self.orientation_mode and self.selected_label >= 0:
                # Calculate angle based on mouse position relative to center
                label = self.labels[self.selected_label]
                center_x = label['x_center'] * self.image_width
                center_y = label['y_center'] * self.image_height
                
                dx = pos.x() - center_x
                dy = pos.y() - center_y
                angle = math.degrees(math.atan2(dy, dx))
                
                # Update label angle
                label['angle'] = angle
                self.rotation_slider.setValue(int(angle))
                self.rotation_label.setText(f"{angle:.1f}°")
                self.angle_edit.setText(f"{angle:.6f}")
                
                self.update_display()
                self.update_labels_list()
    
    def check_near_existing_points(self, pos):
        """Check if click is near an existing label point for editing"""
        for i, label in enumerate(self.labels):
            x_center = label['x_center'] * self.image_width
            y_center = label['y_center'] * self.image_height
            width = label['width'] * self.image_width
            height = label['height'] * self.image_height
            angle = math.radians(label['angle'])
            
            # Calculate corner points
            corners = self.calculate_rotated_corners(x_center, y_center, width, height, angle)
            
            # Check distance to each corner and center
            points = corners + [(x_center, y_center)]
            for point in points:
                distance = math.sqrt((pos.x() - point[0])**2 + (pos.y() - point[1])**2)
                if distance < 10:  # 10 pixel threshold
                    self.selected_label = i
                    self.update_labels_list()
                    self.set_edit_mode()
                    return
    
    def calculate_rotated_corners(self, cx, cy, w, h, angle):
        """Calculate coordinates of rotated rectangle corners"""
        corners = []
        half_w, half_h = w / 2, h / 2
        
        # Original corners relative to center
        relative_corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # Rotate each corner
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for dx, dy in relative_corners:
            x_rot = dx * cos_a - dy * sin_a
            y_rot = dx * sin_a + dy * cos_a
            corners.append((cx + x_rot, cy + y_rot))
        
        return corners
    
    def export_rotated_yolo(self):
        """Export labels in rotated YOLO format"""
        if not self.image_dir:
            return
        
        export_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not export_dir:
            return
        
        # Create annotations directory
        ann_dir = os.path.join(export_dir, "labels")
        img_dir = os.path.join(export_dir, "images")
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(img_dir, exist_ok=True)
        
        # Create dataset.yaml file
        yaml_content = f"""path: {export_dir}
train: images/train
val: images/val

nc: {len(self.classes)}
names: {self.classes}
"""
        
        with open(os.path.join(export_dir, "dataset.yaml"), 'w') as f:
            f.write(yaml_content)
        
        # Export all images and labels
        for i, image_file in enumerate(self.image_files):
            # Copy image
            src_path = os.path.join(self.image_dir, image_file)
            dst_path = os.path.join(img_dir, image_file)
            
            # Read and save labels for this image
            label_path = os.path.join(self.image_dir, 
                                     os.path.splitext(image_file)[0] + '.txt')
            
            if os.path.exists(label_path):
                dst_label_path = os.path.join(ann_dir, 
                                            os.path.splitext(image_file)[0] + '.txt')
                os.system(f'copy "{src_path}" "{dst_path}"' if os.name == 'nt' 
                         else f'cp "{src_path}" "{dst_path}"')
                os.system(f'copy "{label_path}" "{dst_label_path}"' if os.name == 'nt' 
                         else f'cp "{label_path}" "{dst_label_path}"')
        
        QMessageBox.information(self, "Export Complete", 
                              f"Dataset exported to {export_dir}\n\n"
                              f"Format: Rotated YOLO\n"
                              f"Classes: {len(self.classes)}\n"
                              f"Images: {len(self.image_files)}")

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Set dark theme
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    app.setPalette(dark_palette)
    
    window = RotatedYOLOLabelTool()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()