"""
Модуль детекции нарушений с использованием YOLO
"""
import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

class ViolationDetector:
    """Детектор нарушений дисциплины"""
    
    def __init__(self, model_path, conf_threshold=0.2):
        """
        Args:
            model_path: путь к обученной YOLO модели
            conf_threshold: порог уверенности для детекции
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.target_classes = [0, 1, 2, 3]
        self.class_names_map = {
            0: 'sleeping', 
            1: 'phone', 
            2: 'food', 
            3: 'bottle'
        }
        
    def detect_frame(self, frame, draw_boxes=False):
        """
        Детектирует нарушения в кадре
        
        Args:
            frame: изображение (np.ndarray)
            draw_boxes: рисовать ли боксы (default: False, пусть рисует app.py)
            
        Returns:
            Словарь с обнаруженными нарушениями и аннотированное изображение
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        detections = {}
        annotated_frame = frame.copy()
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names_map.get(cls_id, str(cls_id))
            conf = float(box.conf[0])
            
            if cls_id in self.target_classes:
                # Сохраняем обнаруженные классы
                if cls_name not in detections:
                    detections[cls_name] = []
                detections[cls_name].append({
                    'conf': conf,
                    'box': box.xyxy[0].cpu().numpy()
                })
                
                # Рисуем ТОЛЬКО если явно указано
                if draw_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return detections, annotated_frame
    
    def draw_detections(self, frame, detections, violation_names=None):
        """
        Рисует боксы только для указанных нарушений
        
        Args:
            frame: исходное изображение
            detections: словарь обнаруженных объектов из detect_frame()
            violation_names: set названий нарушений, которые нужно отрисовать
                           Если None, рисует все
            
        Returns:
            Аннотированное изображение с боксами
        """
        annotated_frame = frame.copy()
        
        if violation_names is None:
            violation_names = set(detections.keys())
        
        for class_name, boxes_list in detections.items():
            # Рисуем только если это нарушение из confirmed_violations
            if class_name in violation_names:
                for box_info in boxes_list:
                    x1, y1, x2, y2 = map(int, box_info['box'])
                    conf = box_info['conf']
                    
                    # Зеленый цвет для подтвержденных нарушений
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return annotated_frame
    
    def get_class_names(self):
        """Возвращает список названий классов"""
        return list(self.class_names_map.values())
