"""
Models package for knowledge distillation
"""

from .teacher_model import TeacherModel
from .student_model import StudentModel

__all__ = ['TeacherModel', 'StudentModel']
