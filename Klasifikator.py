import numpy as np


class Klasifikator:
    def __init__(self, stevilo_sosedov=3, nacin_razdalje='evklidska'):
        self.stevilo_sosedov = stevilo_sosedov
        self.nacin_razdalje = nacin_razdalje

