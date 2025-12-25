"""
Функции для правильной детекции нарушений и распознавания лиц
Основано на коде от пользователя
"""
import os
import cv2
import time
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from insightface.app import FaceAnalysis


def process_frame_for_detection_correct(current_time, detections_in_frame, sleep_start_time, sleep_buffer):
    """
    Правильная обработка кадра для получения confirmed_violations.
    
    Логика:
    - 'sleeping': требует sleep_buffer секунд подтверждения
    - остальные: подтверждаются сразу
    
    Args:
        current_time: текущее время
        detections_in_frame: set названий обнаруженных классов
        sleep_start_time: время начала обнаружения сна (или None)
        sleep_buffer: буфер подтверждения сна в секундах
    
    Returns:
        (confirmed_violations, new_sleep_start_time, new_last_detection_time)
    """
    confirmed_violations = set()
    new_sleep_start_time = sleep_start_time
    new_last_detection_time = None
    
    sleep_detected = 'sleeping' in detections_in_frame
    
    # ЛОГИКА БУФЕРА СНА
    if sleep_detected:
        if new_sleep_start_time is None:
            new_sleep_start_time = current_time
        
        if (current_time - new_sleep_start_time) >= sleep_buffer:
            confirmed_violations.add('sleeping')
            new_last_detection_time = current_time
    else:
        new_sleep_start_time = None
    
    # ДОБАВЛЕНИЕ ОСТАЛЬНЫХ НАРУШЕНИЙ
    for class_name in detections_in_frame:
        if class_name != 'sleeping':
            confirmed_violations.add(class_name)
            new_last_detection_time = current_time
    
    return confirmed_violations, new_sleep_start_time, new_last_detection_time


def draw_detections_with_boxes(frame, detections_in_frame, class_names_map, detections_dict=None):
    """
    Рисует красные боксы для всех обнаруженных объектов.
    
    Args:
        frame: изображение
        detections_in_frame: set названий обнаруженных классов
        class_names_map: словарь маппинга ID->Name
        detections_dict: полный словарь с координатами (опционально для GREEN боксов)
    
    Returns:
        аннотированный кадр
    """
    annotated_frame = frame.copy()
    
    # Если есть полный словарь детекций с координатами
    if detections_dict:
        for class_name, boxes_list in detections_dict.items():
            if class_name in detections_in_frame:
                for box_info in boxes_list:
                    x1, y1, x2, y2 = map(int, box_info['box'])
                    conf = box_info['conf']
                    # GREEN для подтвержденных
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return annotated_frame


def draw_sleep_indicator(frame, sleep_detected, sleep_start_time, current_time, sleep_buffer):
    """
    Рисует индикатор буфера сна.
    
    Args:
        frame: изображение
        sleep_detected: обнаружен ли сон
        sleep_start_time: время начала сна
        current_time: текущее время
        sleep_buffer: буфер подтверждения
    
    Returns:
        аннотированный кадр
    """
    if sleep_detected and sleep_start_time is not None:
        time_elapsed = current_time - sleep_start_time
        if time_elapsed >= sleep_buffer:
            cv2.putText(frame, "SLEEP CONFIRMED", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            time_left = sleep_buffer - time_elapsed
            cv2.putText(frame, f"Sleep Buffer: {time_left:.1f}s", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
    
    return frame


def load_face_resources():
    """Инициализирует InsightFace и базу данных"""
    try:
        app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception:
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

    db = {}
    if os.path.exists("students.pkl"):
        with open("students.pkl", "rb") as f:
            db = pickle.load(f)
    return app, db


def analyze_video_segment(video_path, app, db, face_similarity=0.5, faces_dir="monitor_output/faces"):
    """
    Проходит по видеофрагменту и ищет лучшее совпадение с базой лиц.
    """
    os.makedirs(faces_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    best_name = "Не опознан"
    max_score = 0.0
    best_face_img = None
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % 5 == 0:
            faces = app.get(frame)
            for face in faces:
                local_name = "Unknown"
                local_score = 0.0
                
                if db:
                    for db_name, db_emb in db.items():
                        sim = np.dot(face.embedding, db_emb) / (np.linalg.norm(face.embedding) * np.linalg.norm(db_emb))
                        if sim > local_score:
                            local_score = sim
                            local_name = db_name
                
                if local_score > max_score:
                    max_score = local_score
                    if local_score >= face_similarity:
                        best_name = local_name
                    else:
                        best_name = "Не опознан"
                    
                    box = face.bbox.astype(int)
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, box[0]), max(0, box[1])
                    x2, y2 = min(w, box[2]), min(h, box[3])
                    best_face_img = frame[y1:y2, x1:x2].copy()
        
        frame_idx += 1
    cap.release()
    
    saved_face_path = "Нет лица"
    if best_face_img is not None and best_face_img.size > 0:
        face_filename = f"face_{Path(video_path).stem}.jpg"
        saved_face_path = os.path.join(faces_dir, face_filename)
        os.makedirs(faces_dir, exist_ok=True)
        cv2.imwrite(saved_face_path, best_face_img)
        
    return best_name, max_score, saved_face_path
