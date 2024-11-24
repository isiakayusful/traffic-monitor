import cv2
from ultralytics import YOLO
from utils.preprocess import load_video, draw_roi
from utils.postprocess import draw_boxes
from utils.config import DEVICE, MODEL_PATH, VIDEO_PATH, OUTPUT_PATH

def main():
    # Load model and video
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    cap = load_video(VIDEO_PATH)
    roi_line = 300  # Example ROI line
    
    # Output video writer
    width, height = int(cap.get(3)), int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30.0, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        detections = results.pandas().xyxy[0]
        frame = draw_boxes(frame, detections, model.names)
        frame = draw_roi(frame, roi_line)

        # Show and save output
        cv2.imshow('Traffic Monitoring', frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
