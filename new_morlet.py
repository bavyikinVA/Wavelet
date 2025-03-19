"""
# используется
import ctypes
import datetime
import os
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image
# from PIL.ImImagePlugin import number

from Gram_Shmidt import change_channels
from pipette import pipette
import sys
from tkinter import messagebox as mb
import matplotlib.pyplot as plt
import customtkinter as ctk

class ImageProcessor:
    def __init__(self):
        self.image_path = ""
        self.folder_path = ""
        self.data = []
        self.data_copy = []
        self.num_scale = 0
        self.scales = np.array([], dtype=ctypes.c_double)
        self.result = []
        self.points_max_by_row = []
        self.points_min_by_row = []
        self.points_max_by_column = []
        self.points_min_by_column = []
        self.extremum_row_min_array = []
        self.extremum_row_max_array = []
        self.extremum_col_min_array = []
        self.extremum_col_max_array = []
        self.extremum_plane_max_array = []

    @staticmethod
    def convert_to_png(image_file_path):
        """