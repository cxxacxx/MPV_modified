#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 11:24:24 2021

@author: cxx
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
import simulation
from tqdm import tqdm
from Utils import IO, Algo, Plot
from matplotlib import cm
from scipy.stats import pearsonr
matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

#array = np.load('data/all_joint_angle_mean.npy',allow_pickle=True)
#gait_data = IO.read_joint_angles_with_gait_phases(read_csv=False, data_file_path='data/gait_data.npy')
#gait_data = IO.read_separate_gait_files(read_csv=True, data_file_path='data/separate_gait_data.npy')
IO.write_mean_and_mono_joint_angle()
#IO.write_mean_thigh_angle()
#IO.write_phase_joint_angle()