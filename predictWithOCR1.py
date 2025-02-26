import hydra
import torch
import easyocr
import cv2
import csv
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import re

def validate_ocr_text(ocr_text):
    pattern1 = r"^[A-Z]{2}\d{1,2}[A-Z]{2}\d{4}$"  # AB12DC3456
    pattern2 = r"^[A-Z]{2}\d{1}[A-Z]{3}\d{4}$"    # AB1DCE3456
    return re.match(pattern1, ocr_text) or re.match(pattern2, ocr_text)

def getOCR(im, coors):
    try:
        x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
        im = im[y:h, x:w]
        conf = 0.2

        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        results = reader.readtext(gray)
        ocr, ocr_confidence = "", 0.0

        for result in results:
            text = result[1].replace(" ", "").upper()  
            if validate_ocr_text(text): 
                ocr, ocr_confidence = text, result[2]
                break  
        print(f"OCR Result: {ocr}, Confidence: {ocr_confidence}")
        return ocr, ocr_confidence

    except Exception as e:
        print(f"Error in getOCR: {e}")
        return "", 0.0  


class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

 
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
      log_string += '%gx%g ' % im.shape[2:]  # print string
      self.annotator = self.get_annotator(im0)

      det = preds[idx]
      self.all_outputs.append(det)
      if len(det) == 0:
        return log_string

    # Write CSV header only once
      if not hasattr(self, 'csv_header_written'):
        with open('results.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image', 'OCR Text', 'Confidence'])
        self.csv_header_written = True

      with open('results.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            label = None if self.args.hide_labels else (
                self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
            ocr, ocr_confidence = getOCR(im0, xyxy)
            if validate_ocr_text(ocr) and ocr_confidence > 0.5:  # Only store validated text
                label = ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
                print(f"Image: {self.data_path.stem}, OCR: {ocr}, Confidence: {ocr_confidence}")
                csvwriter.writerow([self.data_path.stem, ocr, ocr_confidence])

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
