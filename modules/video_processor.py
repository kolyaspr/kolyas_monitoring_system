"""
Модуль для обработки видео и записи сегментов нарушений
"""
import os
import cv2
from datetime import datetime
from pathlib import Path
from collections import Counter


class VideoProcessor:
    """Обработчик видео и запись сегментов"""
    
    def __init__(self, buffer_seconds=10, frame_skip=2, 
                 sleep_persistence_seconds=10):
        """
        Args:
            buffer_seconds: буфер записи после исчезновения нарушения
            frame_skip: обрабатывать каждый N-й кадр
            sleep_persistence_seconds: буфер подтверждения сна
        """
        self.buffer_seconds = buffer_seconds
        self.frame_skip = frame_skip
        self.sleep_persistence_seconds = sleep_persistence_seconds
        self.writer = None
        self.recording = False
    
    def setup_output_dirs(self):
        """Создает необходимые директории для выходных файлов"""
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        segments_dir = os.path.join("monitor_output", "segments", today_date)
        faces_dir = os.path.join("monitor_output", "faces", today_date)
        
        os.makedirs(segments_dir, exist_ok=True)
        os.makedirs(faces_dir, exist_ok=True)
        
        return segments_dir, faces_dir
    
    def start_recording(self, output_path, frame_size, fps):
        """
        Начинает запись видео
        
        Args:
            output_path: путь для сохранения видеофайла
            frame_size: размер кадра (width, height)
            fps: частота кадров
        """
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            frame_size
        )
        self.recording = True
    
    def write_frame(self, frame):
        """Записывает кадр в видеофайл"""
        if self.writer and self.recording:
            self.writer.write(frame)
    
    def stop_recording(self):
        """Останавливает запись видео"""
        if self.writer:
            self.writer.release()
            self.writer = None
            self.recording = False
    
    def generate_segment_filename(self):
        """Генерирует имя файла сегмента на основе времени"""
        timestamp = datetime.now().strftime("%H-%M-%S")
        return f"seg_{timestamp}.mp4"
    
    def generate_report(self, segments_data, face_recognizer=None):
        """
        Генерирует отчет о нарушениях в формате как в примере задания.
        segments_data — список словарей, которые лежат в st.session_state.violations_log
        """

        if not segments_data:
            return None

        today_date = datetime.now().strftime("%Y-%m-%d")
        report_dir = "monitor_output"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"report_{today_date}.txt")

    # Примерное время мониторинга: от первого события до последнего
        times = [v.get("time") for v in segments_data if v.get("time")]
        monitoring_start = times[0] if times else "—"
        monitoring_end = times[-1] if times else "—"

    # Подсчеты
        total = len(segments_data)

        def is_identified(name: str) -> bool:
            if not name:
                return False
            low = name.lower()
            return ("не опознан" not in low) and ("обработка" not in low) and ("unknown" not in low)

        identified = sum(1 for v in segments_data if is_identified(v.get("student", "")))
        not_identified = total - identified

        type_counter = Counter()
        for item in segments_data:
            for t in str(item.get("violation", "")).split(","):
                t = t.strip()
                if t:
                    type_counter[t] += 1

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("═" * 72 + "\n")
            f.write("                    ОТЧЁТ О НАРУШЕНИЯХ ДИСЦИПЛИНЫ\n")
            f.write(f"                    Дата: {today_date}\n")
            f.write(f"                    Время мониторинга: {monitoring_start} - {monitoring_end}\n")
            f.write("═" * 72 + "\n\n")

            for i, item in enumerate(segments_data, 1):
            # Время и длительность (если нет — ставим —)
                start_t = item.get("time_start") or item.get("time") or "—"
                end_t = item.get("time_end") or "—"
                dur = item.get("duration_sec")
                dur_txt = f"{int(dur)} сек" if isinstance(dur, (int, float)) else "—"

                viol_type = item.get("violation", "—")
                violator = item.get("student") or "Неизвестный"
                video_path = item.get("path", "—")

            # confidence может быть '80%' или 'N/A' — оставим как есть
                conf = item.get("confidence", "—")
                if conf in (None, "", "N/A"):
                    conf = "—"

            # Фото лица: попробуем найти в item (если ты где-то сохраняешь)
                face_path = item.get("face_path") or "—"

                f.write(f"№{i}. НАРУШЕНИЕ\n")
                f.write("────────────────────────────────────────────────────────────────────────\n")
                f.write(f"  Время:        {start_t} - {end_t} ({dur_txt})\n")
                f.write(f"  Тип:          {viol_type}\n")
                f.write(f"  Нарушитель:   {violator}\n")
                f.write(f"  Видеозапись:  {video_path}\n")
                f.write(f"  Фото лица:    {face_path}\n")
                f.write(f"  Уверенность:  {conf}\n")
                f.write("────────────────────────────────────────────────────────────────────────\n\n")

            f.write("═" * 72 + "\n")
            f.write("                           ИТОГО\n")
            f.write("═" * 72 + "\n")
            f.write(f"  Всего нарушений:              {total}\n")
            f.write(f"  Идентифицировано:             {identified}\n")
            f.write(f"  Не идентифицировано:          {not_identified}\n\n")
            f.write("  По типам:\n")

            for k, cnt in type_counter.items():
                f.write(f"    - {k}:     {cnt}\n")

            f.write("═" * 72 + "\n")

        return report_file
