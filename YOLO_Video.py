from ultralytics import YOLO
import cv2
import math
import os


def video_detection(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    model = YOLO("YOLO-Weights/ppe.pt")
    classNames = [
        "Excavator",
        "Gloves",
        "Helmet",
        "Ladder",
        "Mask",
        "NO-Hardhat",
        "NO-Mask",
        "NO-Safety Vest",
        "Person",
        "SUV",
        "Safety Cone",
        "Safety Vest",
        "bus",
        "dump truck",
        "fire hydrant",
        "glove",
        "goggles",
        "machinery",
        "mini-van",
        "no_glove",
        "no_goggles",
        "no_shoes",
        "sedan",
        "semi",
        "shoes",
        "trailer",
        "truck",
        "truck and trailer",
        "van",
        "vehicle",
        "wheel loader",
    ]

    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True, device=0)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                if class_name in [
                    "NO-Hardhat",
                    "NO-Mask",
                    "NO-Safety Vest",
                    "bus",
                    "dump truck",
                    "fire hydrant",
                    "machinery",
                    "mini-van",
                    "no_glove",
                    "no_goggles",
                    "no_shoes",
                    "sedan",
                    "semi",
                    "trailer",
                    "truck",
                    "truck and trailer",
                    "van",
                    "vehicle",
                    "wheel loader",
                ]:
                    label = f"{class_name}{conf}"
                    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                    c2 = x1 + t_size[0], y1 - t_size[1] - 3
                    if class_name in [
                        "NO-Hardhat",
                        "NO-Mask",
                        "NO-Safety Vest",
                        "bus",
                        "dump truck",
                        "fire hydrant",
                        "machinery",
                        "mini-van",
                        "no_glove",
                        "no_goggles",
                        "no_shoes",
                        "sedan",
                        "semi",
                        "trailer",
                        "truck",
                        "truck and trailer",
                        "van",
                        "vehicle",
                        "wheel loader",
                    ]:
                        color = (85, 45, 255)
                    else:
                        color = (222, 82, 175)
                    if conf > 0.5:
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                        cv2.putText(
                            img,
                            label,
                            (x1, y1 - 2),
                            0,
                            1,
                            [255, 255, 255],
                            thickness=1,
                            lineType=cv2.LINE_AA,
                        )

        out.write(img)
        yield img

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Function to ensure the output directory exists
def ensure_output_dir(output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
