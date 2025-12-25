"""
Модуль распознавания лиц с InsightFace
"""
import os
import cv2
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from insightface.app import FaceAnalysis

class FaceRecognizer:
    """Распознаватель лиц из видеосегментов"""
    
    def __init__(self, db_path=None):
        """
        Args:
            db_path: путь к сохраненной базе данных эмбеддингов
        """
        self.app = self._init_face_app()
        self.db = {}
        self.db_path = db_path
        
        if db_path and os.path.exists(db_path):
            self.load_database(db_path)
    
    def _init_face_app(self):
        """Инициализирует InsightFace приложение"""
        try:
            app = FaceAnalysis(
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            app = FaceAnalysis(providers=['CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    
    def load_database(self, db_path):
        """Загружает базу данных лиц"""
        try:
            with open(db_path, "rb") as f:
                self.db = pickle.load(f)
        except Exception as e:
            print(f"Ошибка загрузки БД лиц: {e}")
            self.db = {}
    
    def analyze_video_segment(self, video_path, face_similarity=0.5):
        """
        Анализирует видеосегмент и находит лучшее совпадение в БД
        
        Args:
            video_path: путь к видеофайлу
            face_similarity: порог похожести лица
            
        Returns:
            Кортеж (имя_студента, оценка_сходства, путь_к_сохраненному_лицу)
        """
        cap = cv2.VideoCapture(video_path)
        best_name = "Не опознан"
        max_score = 0.0
        best_face_img = None
        
        # Словарь для подсчета совпадений каждого студента
        student_scores = {}  # {student_name: [scores]}
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Анализируем каждый кадр (или можно каждый 2-й для ускорения)
            if frame_idx % 2 == 0:
                try:
                    faces = self.app.get(frame)
                    for face in faces:
                        if self.db:
                            # Ищем совпадение в БД
                            for db_name, db_emb in self.db.items():
                                sim = self._cosine_similarity(
                                    face.embedding, 
                                    db_emb
                                )
                                
                                # Собираем результаты для каждого студента
                                if db_name not in student_scores:
                                    student_scores[db_name] = []
                                student_scores[db_name].append(sim)
                                
                                # Обновляем лучший глобальный результат
                                if sim > max_score:
                                    max_score = sim
                                    best_name = db_name
                                    
                                    # Вырезаем лицо
                                    box = face.bbox.astype(int)
                                    h, w = frame.shape[:2]
                                    x1, y1 = max(0, box[0]), max(0, box[1])
                                    x2, y2 = min(w, box[2]), min(h, box[3])
                                    best_face_img = frame[y1:y2, x1:x2].copy()
                except Exception:
                    # Пропускаем ошибки обработки кадров
                    pass
            
            frame_idx += 1
        
        cap.release()
        
        # Если есть результаты, выбираем студента с наиболее уверенным совпадением
        if student_scores:
            # Для каждого студента берем МАКСИМАЛЬНОЕ совпадение (лучший кадр)
            best_student = None
            best_avg = 0.0
            
            for student_name, scores in student_scores.items():
                max_sim = max(scores)  # Лучшее совпадение этого студента
                avg_sim = np.mean(scores)  # Среднее совпадение
                
                # Выбираем студента с максимальным лучшим совпадением
                if max_sim > best_avg:
                    best_avg = max_sim
                    best_student = student_name
                    best_name = student_name
                    max_score = max_sim
        
        # Сохраняем фото лица
        saved_face_path = "Нет лица"
        if best_face_img is not None and best_face_img.size > 0:
            faces_dir = os.path.join("monitor_output", "faces", 
                                    datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(faces_dir, exist_ok=True)
            
            face_filename = f"face_{Path(video_path).stem}.jpg"
            saved_face_path = os.path.join(faces_dir, face_filename)
            cv2.imwrite(saved_face_path, best_face_img)
        
        return best_name, max_score, saved_face_path
    
    @staticmethod
    def _cosine_similarity(vec1, vec2):
        """Вычисляет косинусовое сходство между двумя векторами"""
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )
    
    def is_database_available(self):
        """Проверяет наличие базы данных"""
        return len(self.db) > 0
