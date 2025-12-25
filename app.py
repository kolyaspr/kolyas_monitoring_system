"""
Streamlit приложение для мониторинга дисциплины на занятиях
Детектирует нарушения: сон, телефон, еда/напитки
"""
import os
import cv2
import time
import pickle
import tempfile
import numpy as np
import streamlit as st
from datetime import datetime
from pathlib import Path

# Импорт модулей
from modules.detection import ViolationDetector
from modules.face_recognition import FaceRecognizer
from modules.video_processor import VideoProcessor
from modules.detection_logic import (
    process_frame_for_detection_correct,
    draw_detections_with_boxes,
    draw_sleep_indicator,
    load_face_resources,
    analyze_video_segment
)

# ═══════════════════════════════════════════════════════════════════════
# КОНФИГУРАЦИЯ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Монитор Дисциплины",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомный стиль
st.markdown("""
<style>
    .main-title {
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #ecf0f1;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .violation-badge {
        display: inline-block;
        padding: 8px 12px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        margin: 5px;
    }
    .sleeping { background-color: #e74c3c; }
    .phone { background-color: #e67e22; }
    .food { background-color: #3498db; }
    .bottle { background-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# ИНИЦИАЛИЗАЦИЯ SESSION STATE
# ═══════════════════════════════════════════════════════════════════════

if 'detector' not in st.session_state:
    st.session_state.detector = None

if 'face_recognizer' not in st.session_state:
    st.session_state.face_recognizer = None

if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()

if 'violations_log' not in st.session_state:
    st.session_state.violations_log = []

if 'processing' not in st.session_state:
    st.session_state.processing = False

# ═══════════════════════════════════════════════════════════════════════
# ФУНКЦИИ КЭШИРОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_detector(model_path):
    """Кэширует детектор нарушений"""
    return ViolationDetector(model_path)

@st.cache_resource
def load_face_recognizer(db_path):
    """Кэширует распознаватель лиц"""
    return FaceRecognizer(db_path if os.path.exists(db_path) else None)

# ═══════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ОБРАБОТКИ
# ═══════════════════════════════════════════════════════════════════════

def process_frame_for_detection(current_time, detections_in_frame, sleep_start_time, sleep_buffer):
    """
    ПРАВИЛЬНАЯ обработка кадра для получения confirmed_violations.
    Использует логику из standalone скрипта.
    
    Args:
        current_time: текущее время
        detections_in_frame: set названий обнаруженных классов {'sleeping', 'phone', ...}
        sleep_start_time: время начала обнаружения сна
        sleep_buffer: буфер подтверждения сна в секундах
    
    Returns:
        (confirmed_violations, new_sleep_start_time, new_last_detection_time)
    """
    return process_frame_for_detection_correct(current_time, detections_in_frame, sleep_start_time, sleep_buffer)


# ═══════════════════════════════════════════════════════════════════════
# ФУНКЦИИ ОБРАБОТКИ ВИДЕО (определены перед использованием)
# ═══════════════════════════════════════════════════════════════════════

def process_webcam(video_container, metrics_container, frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка потока веб-камеры"""
    try:
        st.info("Инициализация веб-камеры... (может занять 3-5 секунд)")
        
        cap = cv2.VideoCapture(0)  # 0 - по умолчанию веб-камера
        
        # Попытка исправить время инициализации
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер
        
        # Проверка что камера открыта
        if not cap.isOpened():
            st.error("Не удалось подключиться к веб-камере. Проверьте её подключение.")
            st.session_state.processing = False
            return
        
        # Попробуем установить параметры камеры для улучшения качества
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создание видеозаписей
        st.session_state.video_processor.setup_output_dirs()
        
        # Метрики
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None
        recording = False
        rec_violations = set()
        current_segment_path = None
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        st.success("Веб-камера готова!")
        st.info("Трансляция с веб-камеры (обновляется в реальном времени)...")
        stop_button = st.button("Остановить трансляцию", key="webcam_stop")
        
        # Обработка до нажатия кнопки Stop или 5 минут (9000 кадров)
        max_frames = 9000
        
        while cap.isOpened() and frame_count < max_frames and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("Ошибка при чтении с веб-камеры")
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()  # Инициализируем на каждой итерации
            
            # Детекция
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                
                # Обновление метрик
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                # ПРАВИЛЬНАЯ ЛОГИКА: получаем подтвержденные нарушения
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                
                # ─── ЛОГИКА ЗАПИСИ ВИДЕО ───
                if confirmed_violations:
                    if not recording:
                        # НАЧАЛО записи
                        segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                        filename = st.session_state.video_processor.generate_segment_filename()
                        current_segment_path = os.path.join(segments_dir, filename)
                        
                        st.session_state.video_processor.start_recording(
                            current_segment_path,
                            (width, height),
                            fps
                        )
                        recording = True
                        rec_violations = set(confirmed_violations)
                    else:
                        # Обновляем типы нарушений (объединяем с уже записанными)
                        rec_violations.update(confirmed_violations)
            
            # ─── ВИЗУАЛИЗАЦИЯ НА КАДРЕ ───
            # 1. Рисуем GREEN боксы ТОЛЬКО для подтвержденных нарушений
            if last_detections and last_confirmed_violations:
                # Пересечение: какие нарушения одновременно обнаружены И подтверждены
                violations_to_draw = last_confirmed_violations & set(last_detections.keys())
                for class_name in violations_to_draw:
                    if class_name in last_detections:
                        boxes_list = last_detections[class_name]
                        for box_info in boxes_list:
                            x1, y1, x2, y2 = map(int, box_info['box'])
                            conf = box_info['conf']
                            # GREEN для подтвержденных
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"{class_name} {conf:.2f}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 2. Индикатор буфера сна
            if 'sleeping' in last_detections:
                if sleep_start_time is not None:
                    time_elapsed = current_time - sleep_start_time
                    if time_elapsed >= sleep_buffer:
                        # Сон подтвержден
                        cv2.putText(annotated_frame, "SLEEP CONFIRMED", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Ждем буфера
                        time_left = sleep_buffer - time_elapsed
                        cv2.putText(annotated_frame, f"Sleep Buffer: {time_left:.1f}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # 3. Красный индикатор записи ТОЛЬКО при нарушениях
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            # ─── ПРОВЕРКА ОКОНЧАНИЯ ЗАПИСИ ───
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    # КОНЕЦ записи
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    
                    # Логируем нарушение
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(sorted(rec_violations)),  # Сортируем для консистентности
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
                    last_confirmed_violations = set()  # Сбрасываем
            
            # Отображение (каждый кадр)
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано кадров: {metrics['total_frames']} | "
                                     f"Нарушений: {len(st.session_state.violations_log)}")
            
            # Примерный прогресс
            progress = min(frame_count / max_frames, 1.0)
            progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        # Анализ лиц ПОСЛЕ завершения обработки
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False
    
    except Exception as e:
        st.error(f"Ошибка при обработке веб-камеры: {e}")
        st.session_state.processing = False

def process_video_file(video_path, video_container, metrics_container, frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка видеофайла"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Создание видеозаписей
        st.session_state.video_processor.setup_output_dirs()
        
        # Метрики
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None  # Время когда закончилась запись
        recording = False
        writer = None
        rec_violations = set()
        current_segment_path = None
        
        # Сохраняем последние обнаруженные нарушения для рисования на всех кадрах
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()  # Инициализируем на каждой итерации
            
            # ─── ДЕТЕКЦИЯ (каждый frame_skip кадр) ───
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                
                # Обновление метрик
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                # ПРАВИЛЬНАЯ ЛОГИКА: получаем подтвержденные нарушения
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                
                # Сохраняем для визуализации
                last_detections = detections
                last_confirmed_violations = confirmed_violations
                
                # Обновляем время последней детекции если есть подтвержденные нарушения
                if confirmed_violations and detection_time:
                    last_detection_time = detection_time
            
            # ─── ЛОГИКА ЗАПИСИ ───
            if confirmed_violations:
                if not recording:
                    # НАЧАЛО записи
                    segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                    filename = st.session_state.video_processor.generate_segment_filename()
                    current_segment_path = os.path.join(segments_dir, filename)
                    
                    st.session_state.video_processor.start_recording(
                        current_segment_path,
                        (width, height),
                        fps
                    )
                    recording = True
                    rec_violations = set(confirmed_violations)
                else:
                    # Обновляем типы нарушений (объединяем с уже записанными)
                    rec_violations.update(confirmed_violations)
            
            # ─── ВИЗУАЛИЗАЦИЯ НА КАДРЕ ───
            # 1. Рисуем RED боксы для ВСЕХ обнаруженных объектов на этом кадре
            if last_detections:
                for class_name, boxes_list in last_detections.items():
                    for box_info in boxes_list:
                        x1, y1, x2, y2 = map(int, box_info['box'])
                        conf = box_info['conf']
                        # RED для всех обнаруженных
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 2. Индикатор буфера сна
            if 'sleeping' in last_detections:
                if sleep_start_time is not None:
                    time_elapsed = current_time - sleep_start_time
                    if time_elapsed >= sleep_buffer:
                        # Сон подтвержден
                        cv2.putText(annotated_frame, "SLEEP CONFIRMED", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else:
                        # Ждем буфера
                        time_left = sleep_buffer - time_elapsed
                        cv2.putText(annotated_frame, f"Sleep Buffer: {time_left:.1f}s", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            
            # Красный индикатор записи
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            # ─── ПРОВЕРКА ОКОНЧАНИЯ ЗАПИСИ ───
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    # КОНЕЦ записи
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    
                    # Логируем нарушение
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(sorted(rec_violations)),  # Сортируем для консистентности
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
                    last_confirmed_violations = set()  # Сбрасываем
            
            # ─── ОТОБРАЖЕНИЕ ───
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано: {metrics['total_frames']} кадров | "
                                     f"Нарушений: {len(st.session_state.violations_log)}")
            
            progress = min(frame_count / (cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1.0)
            progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        # Анализ лиц ПОСЛЕ завершения обработки всего видео
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        # Обновляем ТОЛЬКО если получили валидный результат
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                # Если БД лиц не загружена, отмечаем все как "Не опознан"
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False  # Сбрасываем флаг после завершения
    
    except Exception as e:
        st.error(f"Ошибка при обработке видео: {e}")
        st.session_state.processing = False  # Сбрасываем флаг при ошибке тоже

def process_video_url(url, video_container, metrics_container, frame_skip=2, buffer_seconds=10, sleep_buffer=10, face_db_path="students.pkl", face_similarity=0.5):
    """Обработка видеопотока с URL"""
    try:
        # Попытка несколько способов открыть поток
        st.info(f"Подключение к потоку: {url}")
        
        cap = cv2.VideoCapture(url)
        
        # Проверка что поток открыт с более подробной диагностикой
        if not cap.isOpened():
            st.error("Ошибка подключения к потоку!")
            st.info("""
            **Возможные причины:**
            - Неправильный URL (проверьте формат)
            - Потоковый сервис недоступен или требует аутентификации
            - Сеть недоступна или слабое соединение
            
            **Поддерживаемые форматы URL:**
            - RTSP потоки: `rtsp://...`
            - HTTP потоки: `http://... или https://...`
            - Файловые потоки: `/path/to/video.mp4`
            - Номер камеры: `0` (веб-камера)
            
            **Примеры:**
            - `rtsp://admin:password@192.168.1.100:554/stream`
            - `http://example.com/video.m3u8`
            """)
            st.session_state.processing = False
            return
        
        # Пытаемся получить параметры, если не получается - используем значения по умолчанию
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width == 0:
            width = 1280
        if height == 0:
            height = 720
        
        st.success(f"Подключено! Разрешение: {width}x{height}@{fps}fps")
        
        # Создание видеозаписей
        st.session_state.video_processor.setup_output_dirs()
        
        # Метрики
        metrics = {
            'total_frames': 0,
            'violations': {},
            'recording': False,
            'frames_processed': 0
        }
        
        frame_count = 0
        sleep_start_time = None
        last_detection_time = None
        last_recording_end_time = None
        recording = False
        rec_violations = set()
        current_segment_path = None
        last_detections = {}
        last_confirmed_violations = set()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        st.info("Обработка видеопотока (нажмите Stop для завершения)...")
        
        # Для потоков обрабатываем ограниченное количество кадров или пока не нажмут Stop
        max_frames = 3000  # ~100 секунд на 30 FPS
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            metrics['total_frames'] = frame_count
            
            annotated_frame = frame.copy()
            current_time = time.time()
            confirmed_violations = set()  # Инициализируем на каждой итерации
            
            # Детекция каждые frame_skip кадров
            if frame_count % frame_skip == 0:
                detections, _ = st.session_state.detector.detect_frame(frame, draw_boxes=False)
                last_detections = detections
                
                # Обновление метрик
                for class_name in detections:
                    if class_name not in metrics['violations']:
                        metrics['violations'][class_name] = 0
                    metrics['violations'][class_name] += 1
                
                # CONFIRM: Получение подтвержденных нарушений с учетом буфера сна
                confirmed_violations, sleep_start_time, detection_time = process_frame_for_detection(
                    current_time, set(detections.keys()), sleep_start_time, sleep_buffer
                )
                last_confirmed_violations = confirmed_violations
                if confirmed_violations and detection_time:
                    last_detection_time = detection_time
                
                # Логика управления записью
                if confirmed_violations and not recording:
                    # НАЧАЛО записи
                    segments_dir, _ = st.session_state.video_processor.setup_output_dirs()
                    filename = st.session_state.video_processor.generate_segment_filename()
                    current_segment_path = os.path.join(segments_dir, filename)
                    
                    st.session_state.video_processor.start_recording(
                        current_segment_path,
                        (width, height),
                        fps
                    )
                    recording = True
                    rec_violations = set(confirmed_violations)
                
                # Обновляем типы нарушений если запись уже идет
                if confirmed_violations and recording:
                    rec_violations.update(confirmed_violations)
            
            # VISUALIZE: Рисуем боксы для подтвержденных нарушений
            if last_detections and last_confirmed_violations:
                violations_to_draw = last_confirmed_violations & set(last_detections.keys())
                if violations_to_draw:
                    annotated_frame = st.session_state.detector.draw_detections(
                        annotated_frame, last_detections, violations_to_draw
                    )
            
            # Добавляем индикатор записи (красный круг)
            if recording:
                cv2.circle(annotated_frame, (30, 30), 10, (0, 0, 255), -1)
                st.session_state.video_processor.write_frame(annotated_frame)
            
            # Проверка окончания записи
            if recording and last_detection_time:
                if (current_time - last_detection_time) > buffer_seconds:
                    st.session_state.video_processor.stop_recording()
                    recording = False
                    last_recording_end_time = current_time
                    
                    st.session_state.violations_log.append({
                        'path': current_segment_path,
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'violation': ", ".join(rec_violations),
                        'student': 'Обработка...',
                        'confidence': 'N/A'
                    })
            

            
            # Отображение (каждый кадр)
            frame_placeholder.image(annotated_frame, channels="BGR")
            metrics_placeholder.write(f"Обработано кадров: {metrics['total_frames']} | "
                                     f"Нарушений: {len(st.session_state.violations_log)}")
            
            # Прогресс (примерный для потока)
            progress = min(frame_count / max_frames, 1.0)
            progress_bar.progress(progress)
        
        cap.release()
        if recording:
            st.session_state.video_processor.stop_recording()
        
        # Анализ лиц ПОСЛЕ завершения обработки всего видео
        if st.session_state.violations_log:
            st.info("Анализ лиц в обнаруженных нарушениях...")
            
            if st.session_state.face_recognizer is None and os.path.exists(face_db_path):
                st.session_state.face_recognizer = load_face_recognizer(face_db_path)
            
            if st.session_state.face_recognizer and st.session_state.face_recognizer.is_database_available():
                progress_face = st.progress(0)
                face_status = st.empty()
                
                violations_to_process = [
                    i for i, v in enumerate(st.session_state.violations_log) 
                    if v['student'] == 'Обработка...'
                ]
                
                for idx, i in enumerate(violations_to_process):
                    face_status.text(f"Обработка нарушения {idx + 1}/{len(violations_to_process)}...")
                    violation_path = st.session_state.violations_log[i]['path']
                    
                    try:
                        name, score, face_path = st.session_state.face_recognizer.analyze_video_segment(
                            violation_path,
                            face_similarity=face_similarity
                        )
                        st.session_state.violations_log[i]['student'] = name
                        st.session_state.violations_log[i]['confidence'] = f"{score:.0%}"
                    except Exception as e:
                        st.error(f"Ошибка при анализе {Path(violation_path).name}: {str(e)}")
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Ошибка анализа"
                    
                    progress_face.progress((idx + 1) / len(violations_to_process))
                
                face_status.empty()
                progress_face.empty()
            else:
                for i, violation in enumerate(st.session_state.violations_log):
                    if violation['student'] == 'Обработка...':
                        st.session_state.violations_log[i]['student'] = "Не опознан"
                        st.session_state.violations_log[i]['confidence'] = "Нет БД"
        
        st.success(f"Обработка завершена! Обнаружено {len(st.session_state.violations_log)} нарушений.")
        st.session_state.processing = False
    
    except Exception as e:
        st.error(f"Ошибка при обработке потока: {e}")
        st.session_state.processing = False

def process_violations_data(violations_log):
    """Обработка данных нарушений для анализа"""
    import pandas as pd
    return pd.DataFrame(violations_log) if violations_log else None

def generate_report(violations_log, face_db_path):
    """Генерирует текстовый отчет"""
    try:
        if not st.session_state.face_recognizer and os.path.exists(face_db_path):
            st.session_state.face_recognizer = load_face_recognizer(face_db_path)
        
        return st.session_state.video_processor.generate_report(
            violations_log,
            st.session_state.face_recognizer
        )
    except Exception as e:
        st.error(f"Ошибка при генерации отчета: {e}")
        return None

# ═══════════════════════════════════════════════════════════════════════
# ОСНОВНОЙ ИНТЕРФЕЙС
# ═══════════════════════════════════════════════════════════════════════

st.markdown('<div class="main-title">Система Мониторинга Дисциплины</div>', 
            unsafe_allow_html=True)

# Боковая панель с настройками
st.sidebar.header("")

# Автоматическая загрузка модели best.pt
model_path = "best.pt"
if st.session_state.detector is None and os.path.exists(model_path):
    st.session_state.detector = load_detector(model_path)

# Параметры детекции
st.sidebar.subheader("Параметры детекции")
conf_threshold = st.sidebar.slider(
    "Порог уверенности",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Чем выше - тем строже критерии"
)

frame_skip = st.sidebar.slider(
    "Пропуск кадров",
    min_value=1,
    max_value=10,
    value=2,
    help="Обрабатывать каждый N-й кадр (для скорости)"
)

buffer_seconds = st.sidebar.slider(
    "Буфер записи (сек)",
    min_value=5,
    max_value=30,
    value=10,
    help="Сколько секунд писать после исчезновения нарушения"
)

sleep_buffer = st.sidebar.slider(
    "Буфер сна (сек)",
    min_value=5,
    max_value=30,
    value=10,
    help="Таймер подтверждения сна перед записью"
)

# Параметры распознавания лиц
st.sidebar.subheader("Распознавание лиц")
face_db_path = "students.pkl"
face_similarity = st.sidebar.slider(
    "Порог сходства лица",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Минимальное сходство для опознания"
)

if st.session_state.detector is None:
    st.warning("Модель не загружена. Проверьте наличие файла best.pt")
else:
    # Основные вкладки
    tab1, tab2 = st.tabs([
        "Обработка видео",
        "Статистика"
    ])
    
    # ═══════════════════════════════════════════════════════════════════════
    # ВКЛАДКА 1: ОБРАБОТКА ВИДЕО
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab1:
        st.header("Обработка видео")
        
        col_source, col_options = st.columns([2, 1])
        
        with col_source:
            video_source = st.radio(
                "Выберите источник видео:",
                ["Веб-камера", "Видеофайл", "URL потока"],
                horizontal=True
            )
        
        with col_options:
            if st.button("Начать обработку", key="process_btn", disabled=st.session_state.processing):
                if not st.session_state.processing:
                    st.session_state.processing = True
        
        # Контейнер для видео
        video_container = st.container()
        metrics_container = st.container()
        
        if st.session_state.processing:
            
            if video_source == "Веб-камера":
                process_webcam(video_container, metrics_container,
                             frame_skip, buffer_seconds, sleep_buffer,
                             face_db_path, face_similarity)
            
            elif video_source == "Видеофайл":
                video_file = st.file_uploader(
                    "Загрузите видеофайл",
                    type=['mp4', 'avi', 'mov', 'mkv']
                )
                if video_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_file.read())
                        process_video_file(tmp.name, video_container, metrics_container, 
                                         frame_skip, buffer_seconds, sleep_buffer, 
                                         face_db_path, face_similarity)
            
            elif video_source == "URL потока":
                url = st.text_input("Введите URL видеопотока:")
                if url:
                    process_video_url(url, video_container, metrics_container,
                                     frame_skip, buffer_seconds, sleep_buffer,
                                     face_db_path, face_similarity)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ВКЛАДКА 2: СТАТИСТИКА И ЖУРНАЛ
    # ═══════════════════════════════════════════════════════════════════════
    
    with tab2:
        st.header("Статистика нарушений")
        
        if st.session_state.violations_log:
            violations_df = process_violations_data(st.session_state.violations_log)
            
            # Счетчики
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Всего нарушений", len(st.session_state.violations_log))
            
            with col2:
                sleeping_count = sum(
                    1 for v in st.session_state.violations_log 
                    if 'sleeping' in v['violation'].lower()
                )
                st.metric("Сон", sleeping_count)
            
            with col3:
                phone_count = sum(
                    1 for v in st.session_state.violations_log 
                    if 'phone' in v['violation'].lower()
                )
                st.metric("Телефон", phone_count)
            
            with col4:
                food_count = sum(
                    1 for v in st.session_state.violations_log 
                    if 'food' in v['violation'].lower() or 
                       'bottle' in v['violation'].lower()
                )
                st.metric("Еда/Напиток", food_count)
            
            # Графики
            st.subheader("Распределение по типам нарушений")
            violation_types = {}
            for v in st.session_state.violations_log:
                for vtype in v['violation'].split(', '):
                    violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            if violation_types:
                import plotly.express as px
                fig = px.bar(
                    x=list(violation_types.keys()),
                    y=list(violation_types.values()),
                    labels={'x': 'Тип нарушения', 'y': 'Количество'},
                    color=['#e74c3c', '#e67e22', '#3498db', '#9b59b6'][:len(violation_types)]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Временная шкала
            st.subheader("Временная шкала нарушений")
            times = [v['time'] for v in st.session_state.violations_log]
            st.write(f"Первое нарушение: {times[0] if times else 'N/A'}")
            st.write(f"Последнее нарушение: {times[-1] if times else 'N/A'}")
            st.write(f"Всего записано: {len(st.session_state.violations_log)} фрагментов")
            
            # Кнопки для управления
            st.divider()
            st.subheader("Журнал нарушений")
            col1, col2 = st.columns([1, 1])
            
            # CSV экспорт
            with col1:
                import pandas as pd
                df = pd.DataFrame(st.session_state.violations_log)
                csv = df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8')
                st.download_button(
                    label="Экспорт CSV",
                    data=csv,
                    file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="csv_download_tab2"
                )
            
            with col2:
                if st.button("Очистить", disabled=st.session_state.processing):
                    if not st.session_state.processing:
                        st.session_state.violations_log = []
                        st.success("Журнал очищен")
                        st.rerun()
            
            st.divider()

            st.divider()
            st.subheader("Отчет")

            if st.button("Сгенерировать отчет", key="gen_report"):
                report_path = generate_report(st.session_state.violations_log, face_db_path)
                if report_path and os.path.exists(report_path):
                    st.success(f"Отчет создан: {report_path}")

                    with open(report_path, "r", encoding="utf-8") as f:
                        st.text(f.read())

                    with open(report_path, "rb") as f:
                        st.download_button(
                            label="Скачать отчет",
                            data=f.read(),
                            file_name=Path(report_path).name,
                            mime="text/plain",
                            key="download_report"
                        )

            
            for i, violation in enumerate(st.session_state.violations_log, 1):
                with st.expander(
                    f"Нарушение #{i} | {violation['time']} | {violation['violation']}"
                ):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write(f"**Время:** {violation['time']}")
                        st.write(f"**Нарушение:** {violation['violation']}")
                        st.write(f"**Файл:** {Path(violation['path']).name}")
                    
                    with col2:
                        st.write(f"**Студент:** {violation.get('student', 'Неизвестно')}")
                        st.write(f"**Уверенность:** {violation.get('confidence', 'N/A')}")
                    
                    if os.path.exists(violation['path']):
                        with open(violation['path'], 'rb') as f:
                            st.download_button(
                                label="Скачать видео",
                                data=f.read(),
                                file_name=Path(violation['path']).name,
                                mime="video/mp4",
                                key=f"download_{i}"
                            )
        
        else:
            st.info("Нет данных для отображения. Обработайте видео сначала.")

if __name__ == "__main__":
    pass
