"""Модули для системы мониторинга дисциплины"""
from .detection import ViolationDetector
from .face_recognition import FaceRecognizer
from .video_processor import VideoProcessor

__all__ = ['ViolationDetector', 'FaceRecognizer', 'VideoProcessor']
