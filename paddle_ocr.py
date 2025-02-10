# import hydra
# import torch
# import cv2
# import csv
# import re
# from ultralytics.yolo.engine.predictor import BasePredictor
# from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
# from ultralytics.yolo.utils.checks import check_imgsz
# from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
# from paddleocr import PaddleOCR


# # Initialize PaddleOCR
# ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

# def validate_ocr_text(ocr_text):
#     pattern1 = r"^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$"  # AB12DC3456
#     pattern2 = r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$"    # AB1DCE3456
#     pattern3 = r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$'  # AB1CD2345
#     return re.match(pattern1, ocr_text) or re.match(pattern2, ocr_text) or re.match(pattern3, ocr_text)

# def getOCR(im, coors):
#     try:
#         x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
#         im = im[y:h, x:w]  # Crop the image to the bounding box
#         conf_threshold = 0.2

#         gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
#         results = ocr_reader.ocr(gray, cls=True)
        
#         ocr, ocr_confidence = "", 0.0

#         for result in results:
#             for line in result:
#                 text, confidence = line[1][0], line[1][1]
#                 text = text.replace(" ", "").upper()  # Clean and normalize OCR text
#                 if validate_ocr_text(text) and confidence > conf_threshold:
#                     ocr, ocr_confidence = text, confidence
#                     break  # Stop after finding the first valid match

#         print(f"OCR Result: {ocr}, Confidence: {ocr_confidence}")
#         return ocr, ocr_confidence
    
#     except Exception as e:
#         print(f"Error in getOCR: {e}")
#         return "", 0.0  # Return default values if any error occurs

# class DetectionPredictor(BasePredictor):

#     def get_annotator(self, img):
#         return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

#     def preprocess(self, img):
#         img = torch.from_numpy(img).to(self.model.device)
#         img = img.half() if self.model.fp16 else img.float()
#         img /= 255  # Normalize
#         return img

#     def postprocess(self, preds, img, orig_img):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         agnostic=self.args.agnostic_nms,
#                                         max_det=self.args.max_det)
        
#         for i, pred in enumerate(preds):
#             shape = orig_img[i].shape if self.webcam else orig_img.shape
#             pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        
#         return preds

#     def write_results(self, idx, preds, batch):
#         p, im, im0 = batch
#         log_string = ""
#         if len(im.shape) == 3:
#             im = im[None]
#         self.seen += 1
#         im0 = im0.copy()
#         frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)
        
#         self.data_path = p
#         self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
#         log_string += '%gx%g ' % im.shape[2:]
#         self.annotator = self.get_annotator(im0)
        
#         det = preds[idx]
#         self.all_outputs.append(det)
#         if len(det) == 0:
#             return log_string

#         # Write CSV header only once
#         if not hasattr(self, 'csv_header_written'):
#             with open('results.csv', 'w', newline='') as csvfile:
#                 csvwriter = csv.writer(csvfile)
#                 csvwriter.writerow(['Image', 'OCR Text', 'Confidence'])
#             self.csv_header_written = True

#         with open('results.csv', 'a', newline='') as csvfile:
#             csvwriter = csv.writer(csvfile)
#             for *xyxy, conf, cls in reversed(det):
#                 c = int(cls)
#                 label = None if self.args.hide_labels else (
#                     self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
#                 ocr, ocr_confidence = getOCR(im0, xyxy)
#                 if validate_ocr_text(ocr) and ocr_confidence > 0.05:
#                     label = ocr
#                     self.annotator.box_label(xyxy, label, color=colors(c, True))
#                     print(f"Image: {self.data_path.stem}, OCR: {ocr}, Confidence: {ocr_confidence}")
#                     csvwriter.writerow([self.data_path.stem, ocr, ocr_confidence])
        
#         return log_string

# @hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
# def predict(cfg):
#     cfg.model = cfg.model or "yolov8n.pt"
#     cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
#     cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
#     predictor = DetectionPredictor(cfg)
#     predictor()

# if __name__ == "__main__":
#     predict()



import hydra
import torch
import cv2
import csv
import threading
import pandas as pd
from paddleocr import PaddleOCR  # ✅ Replacing EasyOCR
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import re
import os

# ✅ Initialize PaddleOCR (English model)
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")  

# CSV File Path
CSV_FILE_PATH = "late_results.csv"

# RTSP Stop Flag
rtsp_running = True

# ✅ Function to validate license plate format
def validate_ocr_text(ocr_text):
    patterns = [
        r"^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$",  # AB12DC3456
        r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$",    # AB1DCE3456
        r"^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$"     # AB1CD2345
    ]
    return any(re.match(pattern, ocr_text) for pattern in patterns)

# ✅ OCR Function (Using PaddleOCR)
def getOCR(im, coors):
    try:
        x, y, w, h = map(int, coors[:4])
        im = im[y:h, x:w]  # Crop the image to the bounding box
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        # ✅ Convert image to PaddleOCR format
        results = ocr_model.ocr(gray, cls=True)

        for line in results:
            for word in line:
                text, confidence = word[1][0], word[1][1]
                text = text.replace(" ", "").upper()  # Normalize OCR text
                if validate_ocr_text(text):
                    return text, confidence

        return "", 0.0
    except Exception as e:
        print(f"Error in getOCR: {e}")
        return "", 0.0  # Return default values if any error occurs

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, 
                                        agnostic=self.args.agnostic_nms, 
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def print_results(self):
        print("✅ Processing completed. Results saved to CSV.")

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""

        if len(im.shape) == 3:
            im = im[None]

        self.seen += 1
        im0 = im0.copy()
        frame = self.dataset.count if self.webcam else getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]

        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        self.all_outputs.append(det)

        if len(det) == 0:
            return log_string

        # ✅ Ensure CSV file exists and has the correct header
        if not os.path.exists(CSV_FILE_PATH) or os.stat(CSV_FILE_PATH).st_size == 0:
            with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['Image', 'License Plate', 'Confidence'])

        with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for *xyxy, conf, cls in reversed(det):
                if conf < 0.50:  # 🔹 Increased confidence threshold to 0.50
                    continue
                
                ocr, ocr_confidence = getOCR(im0, xyxy)
                if validate_ocr_text(ocr) and ocr_confidence > 0.0:
                    csvwriter.writerow([self.data_path.stem, ocr, ocr_confidence])
                    print(f"📌 Detected: {ocr} | Confidence: {ocr_confidence:.2f}")

        return log_string


# ✅ Function to process RTSP stream & save data in real time
def process_rtsp_stream(rtsp_url):
    global rtsp_running
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Failed to open RTSP stream!")
        return

    print(f"🔴 Streaming from: {rtsp_url}")
    detector = DetectionPredictor()

    while cap.isOpened() and rtsp_running:
        ret, frame = cap.read()
        if not ret:
            break

        preds = detector.predictor.model(frame)  # Run YOLOv8 on the frame
        detected_plates = detector.write_results(0, preds, (None, frame, frame))

        # ✅ Save to CSV live
        with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for plate in detected_plates:
                csvwriter.writerow(["RTSP_Frame", plate, "N/A"])  

        # Stop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Stopping RTSP stream...")
            break

    cap.release()
    print("✅ RTSP Stream Processing Complete.")


# ✅ Function to stop RTSP Stream
def stop_rtsp_stream():
    global rtsp_running
    rtsp_running = False
    print("🛑 RTSP stream stopping requested.")


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    mode = input("Choose mode (image/video/rtsp): ").strip().lower()
    
    if mode == "rtsp":
        rtsp_url = input("Enter RTSP Stream URL: ").strip()
        print("🔵 Processing RTSP Stream...")
        rtsp_thread = threading.Thread(target=process_rtsp_stream, args=(rtsp_url,), daemon=True)
        rtsp_thread.start()
        
        input("Press ENTER to stop RTSP stream...")
        stop_rtsp_stream()
        rtsp_thread.join()  # Ensure the thread stops before continuing
    
    else:
        predict()
