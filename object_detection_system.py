return detections
    
def _log_detections(self, frame_number, detections):
    for detection in detections:
        log_entry = {
                'frame': frame_number,
                'class': detection['class'],
                'confidence': detection['confidence'],
                'bbox': detection['bbox'],
                'timestamp': time.time()
            }
        self.log_data.append(log_entry)
    
def _save_performance_report(self):
    """Generate and save a performance report"""
    report_path = 'performance_report.txt'
    
    # Get statistics
    stats = self.detector.get_performance_stats()
    
    with open(report_path, 'w') as f:
        f.write("OBJECT DETECTION PERFORMANCE REPORT\n")
        f.write("==================================\n\n")
        
        f.write(f"Model: {self.model_type}\n")
        f.write(f"Processed frames: {stats['processed_frames']}\n")
        f.write(f"Average FPS: {stats['average_fps']:.2f}\n\n")
        
        f.write("Detection Counts:\n")
        for class_name, count in stats['detection_counts'].items():
            if count > 0:
                f.write(f"  - {class_name}: {count}\n")
        
        f.write("\n")
        
        # Add configuration details
        f.write("Configuration:\n")
        for key, value in self.config.items():
            if key != 'classes' and key != 'anchors':
                f.write(f"  - {key}: {value}\n")
    
    return report_path


def get_default_config():
    """Get default configuration for the system"""
    return {
        'model_type': 'yolov3',
        'input_size': (416, 416),  # (height, width)
        'confidence_threshold': 0.5,
        'iou_threshold': 0.45,
        'max_boxes': 100,
        'classes': [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'stop sign', 'parking meter', 'fire hydrant'
        ],
        'anchors': [
            [[116, 90], [156, 198], [373, 326]],
            [[30, 61], [62, 45], [59, 119]],
            [[10, 13], [16, 30], [33, 23]]
        ],
        'anchor_masks': [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        'weights_path': 'models/yolov3_custom.h5',
        'visualization_enabled': True,
        'save_output': True,
        'output_path': 'output'
    }


if __name__ == "__main__":
    print("Script is starting...")
    import sys
    print("Python version:", sys.version)
    
    try:
        print("Testing TensorFlow...")
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        print("Testing OpenCV...")
        import cv2
        print(f"OpenCV version: {cv2.__version__}")
        
        print("Testing numpy...")
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        print("Testing matplotlib...")
        import matplotlib
        print(f"Matplotlib version: {matplotlib.__version__}")
        
        print("Parsing arguments...")
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Advanced Object Detection System for Self-Driving Cars')
        parser.add_argument('--video', type=str, default='0', 
                          help='Path to video file or camera index (default: 0 for webcam)')
        parser.add_argument('--image', type=str, default='', 
                          help='Path to image file (for single image processing)')
        parser.add_argument('--output', type=str, default='', 
                          help='Path to save output video/image')
        parser.add_argument('--model', type=str, default='yolov3',
                          help='Model type (yolov3, rcnn, fast-rcnn)')
        parser.add_argument('--weights', type=str, default=None,
                           help='Path to weights file (overrides default)')
        parser.add_argument('--conf', type=float, default=0.5,
                           help='Confidence threshold (default: 0.5)')
        args = parser.parse_args()
        print(f"Arguments parsed: {args}")
        
        print("Loading configuration...")
        config = get_default_config()
        
        # Update config based on arguments
        if args.model:
            config['model_type'] = args.model
        if args.weights:
            config['weights_path'] = args.weights
        if args.conf:
            config['confidence_threshold'] = args.conf
            
        print("Configuration loaded.")
        
        # Create output directory if needed
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
        
        print("Initializing detection system...")
        # Initialize the system
        detection_system = AdvancedObjectDetectionSystem(config)
        print("Detection system initialized successfully.")
        
        if args.image:
            print(f"Processing image: {args.image}")
            detection_system.process_image(args.image, display=True, output_path=args.output)
        else:
            print(f"Processing video: {args.video}")
            detection_system.process_video(args.video, display=True, output_video=args.output)
            
    except Exception as e:
        print(f"Error occurred: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
import numpy as np
import tensorflow as tf
import cv2
import time
import argparse
import os
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add
from tensorflow.keras.layers import MaxPooling2D, Concatenate, UpSampling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

class CustomYOLOv3:
    """
    Enhanced YOLOv3 implementation for self-driving car object detection
    with performance optimizations and customized architecture
    """
    
    def __init__(self, config):
        """
        Initialize the custom YOLOv3 model
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.input_size = config['input_size']
        self.num_classes = len(config['classes'])
        self.class_names = config['classes']
        self.anchors = np.array(config['anchors'])
        self.anchor_masks = config['anchor_masks']
        self.confidence_threshold = config['confidence_threshold']
        self.iou_threshold = config['iou_threshold']
        self.max_boxes = config['max_boxes']
        
        # Detection metrics
        self.avg_precision = 0
        self.avg_recall = 0
        self.avg_fps = 0
        self.processed_frames = 0
        
        # Set up visualization
        self.class_colors = self._generate_colors()
        self.fps_buffer = deque(maxlen=30)
        self.detection_counts = {cls: 0 for cls in self.class_names}
        
        # Monitoring
        self.monitoring_data = {
            'fps_history': [],
            'detection_counts_history': [],
            'confidence_scores': []
        }
        
        # Check if weights file exists
        weights_path = self.config.get('weights_path')
        if weights_path:
            print(f"Weights path: {weights_path}")
            print(f"Weights file exists: {os.path.exists(weights_path)}")
        else:
            print("No weights path specified.")
        
        # Set up model
        print("Building model...")
        self.model = self._build_model()
        print("Model built successfully.")
        
        print(f"CustomYOLOv3 initialized with {self.num_classes} classes")
    
    def _build_model(self):
        """Build the enhanced YOLOv3 model architecture"""
        # Input layer
        print("Creating input layer...")
        input_layer = Input(shape=(self.input_size[0], self.input_size[1], 3))
        
        # Darknet-53 backbone with residual connections and optimizations
        print("Building darknet backbone...")
        x = self._darknet53_body(input_layer)
        
        # Detection heads for different scales
        print("Building detection heads...")
        y1, y2, y3 = self._detection_heads(x)
        
        # Create the model
        print("Creating model...")
        model = Model(inputs=input_layer, outputs=[y1, y2, y3])
        
        # Load pre-trained weights if specified and file exists
        weights_path = self.config.get('weights_path')
        if weights_path and os.path.exists(weights_path):
            print(f"Loading pre-trained weights from {weights_path}")
            try:
                model.load_weights(weights_path)
                print("Weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights: {e}")
                print("Continuing with randomly initialized weights.")
        else:
            print("No weights file found or specified. Using randomly initialized weights.")
        
        return model
    
    def _darknet_conv(self, x, filters, size, strides=1, batch_norm=True):
        """Darknet convolutional layer with batch normalization option"""
        padding = 'same' if strides == 1 else 'valid'
        
        if strides > 1:
            x = ZeroPadding2D(((1, 0), (1, 0)))(x)
            
        x = Conv2D(filters=filters, kernel_size=size, 
                  strides=strides, padding=padding,
                  use_bias=not batch_norm, 
                  kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01))(x)
                  
        if batch_norm:
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=0.1)(x)
            
        return x
    
    def _darknet_residual(self, x, filters):
        """Residual block for Darknet"""
        prev = x
        x = self._darknet_conv(x, filters // 2, 1)
        x = self._darknet_conv(x, filters, 3)
        x = Add()([prev, x])
        return x
    
    def _darknet53_body(self, x):
        """Enhanced Darknet-53 backbone with optimizations"""
        # Initial convolution
        x = self._darknet_conv(x, 32, 3)
        
        # Downsample 1: 416x416 -> 208x208
        x = self._darknet_conv(x, 64, 3, strides=2)
        for _ in range(1):
            x = self._darknet_residual(x, 64)
            
        # Downsample 2: 208x208 -> 104x104
        x = self._darknet_conv(x, 128, 3, strides=2)
        for _ in range(2):
            x = self._darknet_residual(x, 128)
            
        # Downsample 3: 104x104 -> 52x52
        x = self._darknet_conv(x, 256, 3, strides=2)
        for _ in range(8):
            x = self._darknet_residual(x, 256)
        route1 = x  # First feature map for detection
            
        # Downsample 4: 52x52 -> 26x26
        x = self._darknet_conv(x, 512, 3, strides=2)
        for _ in range(8):
            x = self._darknet_residual(x, 512)
        route2 = x  # Second feature map for detection
            
        # Downsample 5: 26x26 -> 13x13
        x = self._darknet_conv(x, 1024, 3, strides=2)
        for _ in range(4):
            x = self._darknet_residual(x, 1024)
            
        return [route1, route2, x]
    
    def _detection_heads(self, features):
        """Create detection heads for small, medium, and large objects"""
        route1, route2, x = features
        
        # Detection head for large objects (13x13 grid)
        x = self._darknet_conv(x, 512, 1)
        x = self._darknet_conv(x, 1024, 3)
        x = self._darknet_conv(x, 512, 1)
        x = self._darknet_conv(x, 1024, 3)
        x = self._darknet_conv(x, 512, 1)
        large_branch = self._darknet_conv(x, 1024, 3)
        large_output = Conv2D(len(self.anchor_masks[0]) * (5 + self.num_classes),
                             1, strides=1, padding='same')(large_branch)
        
        # Route for medium objects
        x = self._darknet_conv(x, 256, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, route2])
        
        # Detection head for medium objects (26x26 grid)
        x = self._darknet_conv(x, 256, 1)
        x = self._darknet_conv(x, 512, 3)
        x = self._darknet_conv(x, 256, 1)
        x = self._darknet_conv(x, 512, 3)
        x = self._darknet_conv(x, 256, 1)
        medium_branch = self._darknet_conv(x, 512, 3)
        medium_output = Conv2D(len(self.anchor_masks[1]) * (5 + self.num_classes),
                              1, strides=1, padding='same')(medium_branch)
        
        # Route for small objects
        x = self._darknet_conv(x, 128, 1)
        x = UpSampling2D(2)(x)
        x = Concatenate()([x, route1])
        
        # Detection head for small objects (52x52 grid)
        x = self._darknet_conv(x, 128, 1)
        x = self._darknet_conv(x, 256, 3)
        x = self._darknet_conv(x, 128, 1)
        x = self._darknet_conv(x, 256, 3)
        x = self._darknet_conv(x, 128, 1)
        small_branch = self._darknet_conv(x, 256, 3)
        small_output = Conv2D(len(self.anchor_masks[2]) * (5 + self.num_classes),
                             1, strides=1, padding='same')(small_branch)
        
        return [large_output, medium_output, small_output]
    
    def preprocess_image(self, image):
        """Preprocess image for the network"""
        # Resize to model input size
        image_resized = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        
        # Convert to RGB (OpenCV uses BGR)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_expanded = np.expand_dims(image_norm, axis=0)
        
        return image_expanded
    
    def _process_predictions(self, predictions, image_shape):
        """Process YOLO predictions to get final boxes"""
        # Extract predictions for each scale
        boxes, scores, classes = [], [], []
        
        # Process each YOLO detection head output
        for i, pred in enumerate(predictions):
            # Get anchors for this scale
            anchors = [self.anchors[mask] for mask in self.anchor_masks][i]
            
            # Process grid predictions to bounding boxes
            pred_boxes, pred_scores, pred_classes = self._process_layer(
                pred[0], anchors, image_shape)
            
            boxes.append(pred_boxes)
            scores.append(pred_scores)
            classes.append(pred_classes)
        
        # Combine predictions from all scales
        boxes = tf.concat(boxes, axis=0)
        scores = tf.concat(scores, axis=0)
        classes = tf.concat(classes, axis=0)
        
        # Apply non-max suppression
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, self.max_boxes,
            self.iou_threshold, self.confidence_threshold
        )
        
        # Gather the selected boxes
        boxes = tf.gather(boxes, selected_indices)
        scores = tf.gather(scores, selected_indices)
        classes = tf.gather(classes, selected_indices)
        
        return boxes.numpy(), scores.numpy(), classes.numpy()
    
    def _process_layer(self, layer_predictions, anchors, image_shape):
        """Process a single YOLO layer output"""
        # Get grid dimensions
        grid_shape = tf.shape(layer_predictions)[0:2]
        grid_y, grid_x = tf.meshgrid(tf.range(grid_shape[0]), tf.range(grid_shape[1]))
        grid = tf.stack([grid_x, grid_y], axis=-1)
        grid = tf.expand_dims(tf.cast(grid, tf.float32), axis=2)
        
        # Reshape predictions
        layer_predictions = tf.reshape(
            layer_predictions,
            (grid_shape[0], grid_shape[1], len(anchors), 5 + self.num_classes)
        )
        
        # Split predictions
        box_xy = tf.sigmoid(layer_predictions[..., 0:2])
        box_wh = tf.exp(layer_predictions[..., 2:4])
        objectness = tf.sigmoid(layer_predictions[..., 4:5])
        class_probs = tf.sigmoid(layer_predictions[..., 5:])
        
        # Scale box coordinates
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_shape[::-1], tf.float32)
        box_wh = box_wh * anchors / self.input_size
        
        # Convert to corner coordinates (x1, y1, x2, y2)
        boxes = tf.concat([
            box_xy - box_wh / 2,  # top-left
            box_xy + box_wh / 2   # bottom-right
        ], axis=-1)
        
        # Scale to image size
        boxes = boxes * tf.concat([image_shape[::-1], image_shape[::-1]], axis=-1)
        
        # Get confidence and class probabilities
        scores = objectness * tf.reduce_max(class_probs, axis=-1, keepdims=True)
        classes = tf.argmax(class_probs, axis=-1)
        
        # Reshape outputs
        boxes = tf.reshape(boxes, [-1, 4])
        scores = tf.reshape(scores, [-1])
        classes = tf.reshape(classes, [-1])
        
        return boxes, scores, classes
    
    def detect(self, image):
        """Detect objects in an image"""
        start_time = time.time()
        
        # Save original dimensions for scaling
        original_height, original_width = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        try:
            print(f"Running inference on image with shape {image.shape}")
            predictions = self.model.predict(input_tensor, verbose=0)
            
            # Process predictions
            boxes, scores, class_ids = self._process_predictions(
                predictions, (original_height, original_width))
            
            # Update detection metrics
            self.processed_frames += 1
            for class_id in class_ids:
                class_name = self.class_names[int(class_id)]
                self.detection_counts[class_name] += 1
            
            # Add scores to monitoring data
            self.monitoring_data['confidence_scores'].extend(scores.tolist())
            
            # Calculate FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time
            self.fps_buffer.append(fps)
            avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
            self.avg_fps = avg_fps
            self.monitoring_data['fps_history'].append(fps)
            
            # Prepare detection results
            detections = []
            for i in range(len(boxes)):
                detections.append({
                    'class': self.class_names[int(class_ids[i])],
                    'confidence': float(scores[i]),
                    'bbox': boxes[i].astype(int).tolist(),  # [x1, y1, x2, y2]
                })
            
            # Visualize results on the image
            processed_image = self._visualize_detections(image.copy(), boxes, scores, class_ids)
            
            return processed_image, detections
        
        except Exception as e:
            print(f"Error during detection: {e}")
            import traceback
            traceback.print_exc()
            
            # Return original image if detection fails
            return image.copy(), []
    
    def _visualize_detections(self, image, boxes, scores, class_ids):
        """Draw detections on the image with enhanced visualization"""
        for i in range(len(boxes)):
            # Get detection properties
            x1, y1, x2, y2 = boxes[i].astype(int)
            class_id = int(class_ids[i])
            class_name = self.class_names[class_id]
            confidence = scores[i]
            color = self.class_colors[class_id]
            
            # Clamp coordinates to image boundaries
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            # Draw box with thickness based on confidence
            thickness = max(2, int(confidence * 4))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Create label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"
            
            # Calculate label background size
            font_scale = 0.5
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            
            # Draw label background
            cv2.rectangle(
                image, 
                (x1, y1 - text_height - 5), 
                (x1 + text_width, y1),
                color, -1
            )
            
            # Draw label text (white on colored background)
            cv2.putText(
                image, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
            )
        
        # Add performance metrics to the image
        avg_fps = sum(self.fps_buffer) / max(len(self.fps_buffer), 1)
        metrics_text = f"FPS: {avg_fps:.1f}"
        
        # Count objects
        object_counts = {}
        for i in range(len(class_ids)):
            class_name = self.class_names[int(class_ids[i])]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Add counts to metrics text
        if object_counts:
            counts_str = ", ".join([f"{k}: {v}" for k, v in object_counts.items()])
            metrics_text += f" | Objects: {counts_str}"
        
        # Draw metrics background
        cv2.rectangle(image, (10, 10), (10 + 400, 10 + 30), (0, 0, 0), -1)
        cv2.putText(
            image, metrics_text, (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
        )
        
        return image
    
    def _generate_colors(self):
        """Generate visually distinct colors for classes"""
        # Use HSV colorspace to generate evenly spaced colors
        hsv_tuples = [(x / self.num_classes, 0.8, 1.0) for x in range(self.num_classes)]
        colors = []
        for hsv in hsv_tuples:
            # Convert HSV to RGB
            rgb = tuple(int(x * 255) for x in plt.colormaps['hsv'](hsv[:1])[0][:3])
            colors.append(rgb)
        return colors
    
    def generate_monitoring_dashboard(self, save_path="monitoring_dashboard.png"):
        """Generate a dashboard with performance metrics and visualizations"""
        try:
            fig, ax = plt.subplots(2, 2, figsize=(15, 10))
            
            # FPS history
            if self.monitoring_data['fps_history']:
                ax[0, 0].plot(self.monitoring_data['fps_history'])
                ax[0, 0].set_title('FPS History')
                ax[0, 0].set_xlabel('Frame')
                ax[0, 0].set_ylabel('FPS')
                ax[0, 0].grid(True)
            else:
                ax[0, 0].text(0.5, 0.5, "No FPS data available", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Object detection counts
            labels = []
            counts = []
            for class_name, count in self.detection_counts.items():
                if count > 0:
                    labels.append(class_name)
                    counts.append(count)
            
            if counts:
                ax[0, 1].bar(labels, counts, color='skyblue')
                ax[0, 1].set_title('Object Detection Counts')
                ax[0, 1].set_xlabel('Object Class')
                ax[0, 1].set_ylabel('Count')
                plt.setp(ax[0, 1].get_xticklabels(), rotation=45, ha='right')
            else:
                ax[0, 1].text(0.5, 0.5, "No objects detected", 
                              horizontalalignment='center', verticalalignment='center')
            
            # Confidence score distribution
            if self.monitoring_data['confidence_scores']:
                ax[1, 0].hist(self.monitoring_data['confidence_scores'], bins=20, range=(0, 1))
                ax[1, 0].set_title('Confidence Score Distribution')
                ax[1, 0].set_xlabel('Confidence Score')
                ax[1, 0].set_ylabel('Frequency')
            else:
                ax[1, 0].text(0.5, 0.5, "No confidence score data available", 
                              horizontalalignment='center', verticalalignment='center')
                
            # Processing info
            info_text = (
                f"Total Frames Processed: {self.processed_frames}\n"
                f"Average FPS: {self.avg_fps:.2f}\n"
                f"Objects Detected: {sum(counts) if counts else 0}\n"
                f"Model Input Size: {self.input_size[0]}x{self.input_size[1]}\n"
                f"Confidence Threshold: {self.confidence_threshold}\n"
                f"IoU Threshold: {self.iou_threshold}"
            )
            ax[1, 1].text(0.1, 0.5, info_text, fontsize=12)
            ax[1, 1].set_title('Processing Information')
            ax[1, 1].axis('off')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
            return save_path
        except Exception as e:
            print(f"Error generating monitoring dashboard: {e}")
            return None
    
    def get_performance_stats(self):
        """Get model performance statistics"""
        return {
            'average_fps': self.avg_fps,
            'detection_counts': self.detection_counts,
            'processed_frames': self.processed_frames
        }


class AdvancedObjectDetectionSystem:
    """
    Advanced object detection system for self-driving cars with
    customizable models, visualization interface, and performance analytics
    """
    
    def __init__(self, config):
        """Initialize the object detection system"""
        self.config = config
        
        # Check weights file
        weights_path = config.get('weights_path')
        if weights_path:
            print(f"Looking for weights file at: {weights_path}")
            print(f"File exists: {os.path.exists(weights_path)}")
        
        self.model_type = config.get('model_type', 'yolov3')
        
        # Initialize the appropriate model
        print(f"Initializing {self.model_type} detector...")
        if self.model_type == 'yolov3':
            self.detector = CustomYOLOv3(config)
        elif self.model_type == 'rcnn':
            # Placeholder for R-CNN implementation
            raise NotImplementedError("R-CNN implementation not available yet")
        elif self.model_type == 'fast-rcnn':
            # Placeholder for Fast R-CNN implementation
            raise NotImplementedError("Fast R-CNN implementation not available yet")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Initialize visualization and logging
        self.visualization_enabled = config.get('visualization_enabled', True)
        self.save_output = config.get('save_output', False)
        self.output_path = config.get('output_path', 'output')
        self.log_file = config.get('log_file', 'detection_log.txt')
        
        # Create output directory if it doesn't exist
        if self.save_output and not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print(f"Created output directory: {self.output_path}")
        
        # Initialize logging
        self.log_data = []
        
        print(f"Advanced Object Detection System initialized with {self.model_type}")
    
    def process_video(self, video_path, display=True, output_video=None):
        """Process a video file or camera stream"""
        # Open video source
        if video_path.isdigit():
            video_path = int(video_path)  # Use camera
            
        print(f"Opening video source: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_path}")
            return False
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if needed
        writer = None
        if output_video:
            os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            print(f"Initialized video writer: {output_video}")
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        try:
            print("Starting video processing...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Detect objects
                processed_frame, detections = self.detector.detect(frame)
                
                # Log results
                self._log_detections(frame_count, detections)
                
                # Save or display the processed frame
                if writer:
                    print(f"Writing frame {frame_count} to output video")
                    writer.write(processed_frame)
                    
                if display:
                    # Add progress information
                    if total_frames > 0:
                        progress = frame_count / total_frames * 100
                        cv2.putText(
                            processed_frame, f"Progress: {progress:.1f}%", 
                            (width - 200, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                        )
                        
                    cv2.imshow('Advanced Object Detection', processed_frame)
                    
                    # Process key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        print("Processing stopped by user (ESC key)")
                        break
                
                frame_count += 1
                
                # Print progress every 10 frames
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")
        
        except KeyboardInterrupt:
            print("Processing interrupted by user")
        
        except Exception as e:
            print(f"Error during video processing: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Clean up resources
            print("Cleaning up resources...")
            cap.release()
            if writer:
                print(f"Finalizing video with {frame_count} frames")
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            # Generate and save performance report
            try:
                report_path = self._save_performance_report()
                print(f"Performance report saved to {report_path}")
            except Exception as e:
                print(f"Error saving performance report: {e}")
            
            # Generate monitoring dashboard
            try:
                dashboard_path = self.detector.generate_monitoring_dashboard()
                if dashboard_path:
                    print(f"Monitoring dashboard saved to {dashboard_path}")
            except Exception as e:
                print(f"Error generating monitoring dashboard: {e}")
            
            print(f"Processing complete. Processed {frame_count} frames.")
            
            return True
    
    def process_image(self, image_path, display=True, output_path=None):
        """Process a single image file"""
        # Read image
        print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return False
        
        print(f"Image loaded with shape: {image.shape}")
        
        # Detect objects
        processed_image, detections = self.detector.detect(image)
        
        # Save results if requested
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, processed_image)
            print(f"Processed image saved to {output_path}")
        
        # Display if requested
        if display:
            cv2.imshow('Object Detection Result', processed_image)
            print("Displaying result. Press any key to continue.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection results
        print(f"Detected {len(detections)} objects:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class']} ({detection['confidence']:.2f})")