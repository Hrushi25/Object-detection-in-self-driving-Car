import cv2
import numpy as np
import time
import argparse
import os

# Create directories
os.makedirs('output', exist_ok=True)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Path to video file')
parser.add_argument('--output', help='Path to output video')
parser.add_argument('--conf', default=0.5, type=float, help='Confidence threshold')
args = parser.parse_args()

# Set paths to YOLO files
weights_path = 'yolo_files/yolov3.weights'
config_path = 'yolo_files/yolov3.cfg'
classes_path = 'yolo_files/coco.names'

# Check if files exist
if not os.path.exists(weights_path):
    print(f"ERROR: Weights file not found: {weights_path}")
    exit()
    
if not os.path.exists(config_path):
    print(f"ERROR: Config file not found: {config_path}")
    exit()

if not os.path.exists(classes_path):
    print(f"ERROR: Classes file not found: {classes_path}")
    exit()

# Load YOLO
print("Loading YOLO model...")
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Check if GPU is available and set preference
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("Using CUDA backend")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("Using CPU backend")

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load classes
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Generate colors for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

# Open video
print(f"Opening video: {args.video}")
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video {args.video}")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} total frames")

# Initialize video writer
out = None
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"Writing output to: {args.output}")

frame_count = 0
start_time = time.time()
detection_counts = {class_name: 0 for class_name in classes}

print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break
    
    frame_count += 1
    
    # Only process every other frame for speed
    if frame_count % 2 != 0 and frame_count > 1:
        if out:
            out.write(frame)
        continue
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set input to network
    net.setInput(blob)
    
    # Forward pass through network
    print(f"Processing frame {frame_count}/{total_frames}")
    layer_outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > args.conf:
                # Scale bounding box coordinates to image size
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, args.conf, 0.4)
    
    # Draw bounding boxes
    for i in indices:
        if isinstance(i, list):  # OpenCV 4.5.4 and earlier
            i = i[0]
        
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (int(colors[class_ids[i]][0]), int(colors[class_ids[i]][1]), int(colors[class_ids[i]][2]))
        
        # Update detection count
        detection_counts[label] += 1
        
        # Draw box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add FPS info
    elapsed_time = time.time() - start_time
    fps_text = f"FPS: {frame_count/elapsed_time:.1f} | Progress: {frame_count}/{total_frames}"
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Write to output file
    if out:
        out.write(frame)
    
    # Display frame
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        print("Processing stopped by user")
        break
    
    # Print progress every 30 frames
    if frame_count % 30 == 0:
        print(f"Processed {frame_count}/{total_frames} frames. FPS: {frame_count/elapsed_time:.1f}")

# Print final stats
print(f"\nProcessing complete. {frame_count} frames processed.")
print(f"Average FPS: {frame_count/elapsed_time:.1f}")
print("Detection counts:")
for class_name, count in detection_counts.items():
    if count > 0:
        print(f"  - {class_name}: {count}")

# Clean up
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()