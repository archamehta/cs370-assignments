import cv2
import numpy as np

# Load pre-trained YOLOv4 model
def load_yolov4_model(weights_path, cfg_path):
    model = cv2.dnn.readNet(weights_path, cfg_path)
    return model

# Perform object detection using YOLOv4
def detect_objects_yolov4(image, model, confidence_threshold=0.5, nms_threshold=0.4):
    # Prepare the input image for detection
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input to the model
    model.setInput(blob)

    # Forward pass and get the output layers
    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in model.getUnconnectedOutLayers()]
    outputs = model.forward(output_layers)

    # Process the outputs to extract detections
    detections = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                detections.append((x, y, width, height, class_id, confidence))

    # Apply non-maximum suppression to remove redundant detections
    indices = cv2.dnn.NMSBoxes(detections, confidence_threshold, nms_threshold)
    filtered_detections = [detections[i[0]] for i in indices]

    return filtered_detections

# Function to process video frames
def process_video(input_video_path, output_video_path, model):
    # Open the input video file
    video_capture = cv2.VideoCapture(input_video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame of the video
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Perform object detection on the current frame
        detections = detect_objects_yolov4(frame, model)

        # Visualize the detected objects on the frame
        for x, y, w, h, class_id, confidence in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {class_id} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        video_writer.write(frame)

    # Release the video capture and writer objects
    video_capture.release()
    video_writer.release()

# Main function
def main():
    # Load YOLOv4 model
    weights_path = './Drone Assignment/Drone/yolov3.weights'
    cfg_path = './Drone Assignment/Drone/yolov3.cfg'
    model = load_yolov4_model(weights_path, cfg_path)

    if model.empty():
        print("Error: Failed to load YOLOv3 model.")
        return

    print("YOLOv3 model loaded successfully.")
    # Process input video and save output video
    input_video_path = './Drone Assignment/Drone/video.mp4'
    output_video_path = './Drone Assignment/Drone/video_output.mp4'
    process_video(input_video_path, output_video_path, model)
    
if __name__ == "__main__":
    main()
