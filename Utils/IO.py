import glob
import numpy as np
import matplotlib
import os
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import open3d as o3d

# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
from Utils import Algo, Plot



def save_view_point(vis, viewer_setting_file):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(viewer_setting_file, param)


def load_view_point(vis, viewer_setting_file):
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(viewer_setting_file)
    ctr.convert_from_pinhole_camera_parameters(param)
    return vis

def read_signals(signal_folder):
    signal_names = glob.glob('{}/1*.npy'.format(signal_folder))
    signal_time_vec = obtain_file_time_vec(signal_names)
    frame_num = len(signal_time_vec)
    signal_vec = np.zeros((frame_num, 4*9))
    for i in range(len(signal_names)):
        signal_frame = np.load('{}/{:.3f}.npy'.format(signal_folder, signal_time_vec[i]))
        signal_vec[i] = signal_frame[:-1]
    return signal_vec, signal_time_vec

def read_signals_with_label(signal_folder, save_data= True):
    signal_vec, signal_time_vec = read_signals(signal_folder)
    signal_time_vec -= signal_time_vec[0]
    signal_labels = np.loadtxt('{}/label.csv'.format(signal_folder), delimiter=',')[:, 1]
    signal_labels = signal_labels.astype(np.int)
    signal_vec -= np.mean(signal_vec[signal_labels == 7][10:], axis=0, keepdims=True)
    signal_with_label_vec = np.c_[signal_time_vec.reshape((-1, 1)), signal_labels.reshape((-1, 1)), signal_vec]
    valid_indices = np.logical_and(signal_labels >= 0, signal_labels < 6)
    signal_with_label_vec = signal_with_label_vec[valid_indices] # time_vec, label, imu_data
    if save_data:
        np.save('{}/signal_with_label_vec.npy'.format(signal_folder), signal_with_label_vec)
    return signal_with_label_vec


def obtain_file_time_vec(file_name_list):
    file_time_vec = []
    for file_name in file_name_list:
        file_time = float(os.path.splitext(os.path.basename(file_name))[0])
        file_time_vec.append(file_time)
    file_time_vec = np.around(np.asarray(file_time_vec), decimals=3)
    file_time_vec = np.sort(file_time_vec)
    return file_time_vec


def read_depth_pcd(depth_img_name, pinhole_camera_intrinsic = None):
    if pinhole_camera_intrinsic is None:
        pinhole_camera_intrinsic = Algo.read_cam_intrinsic()
    current_depth = o3d.io.read_image(depth_img_name)
    current_pcd = o3d.geometry.PointCloud.create_from_depth_image(current_depth, pinhole_camera_intrinsic,
                                                                  depth_trunc=5.0, )
    return current_pcd

def read_rgbd_pcd(rgb_img_name, depth_img_name, pinhole_camera_intrinsic = None):
    if pinhole_camera_intrinsic is None:
        pinhole_camera_intrinsic = Algo.read_cam_intrinsic()
    current_color = o3d.io.read_image(rgb_img_name)
    current_depth = o3d.io.read_image(depth_img_name)
    current_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        current_color, current_depth, depth_trunc=5.0, convert_rgb_to_intensity=False)
    current_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        current_rgbd_image, pinhole_camera_intrinsic)  # unit: m
    return current_pcd


def rm_all_files(dst):
    files = glob.glob('{}/*'.format(dst))
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
            
def read_foot_acc(file_path=None):
    '''
    foot_acc_mat: thigh acc x y z
    '''
    data = np.genfromtxt(file_path, delimiter=',')[1:]
    foot_acc_mat = data[:, 1:4]
    return foot_acc_mat          

def read_joint_angles(file_path=None):
    '''
    joint_angle_mat: [absolute angle of left hip in the sagital plane, absolute angle of right hip in the sagital plane]
    absolute angle means the angle is relative to the gravity direction.
    '''
    data = np.genfromtxt(file_path, delimiter=',')[1:]
    pelvis_tilt_sagittal = data[:, [1]]
    '''
    Left hip, knee, ankle; Right hip, knee, ankle.
    '''
    joint_angle_mat = np.zeros((data.shape[0], 6))
    joint_angle_mat[:, [0, 3]] = data[:, [14, 7]] - pelvis_tilt_sagittal
    joint_angle_mat[:, 1:3] = data[:, 17:19]
    joint_angle_mat[:, 4:6] = data[:, 10:12]
    joint_angle_mat[:, 4:] = read_joint_angles_from_goni(file_path, ref_time_vec=data[:, 0])[:, 1:]
    return joint_angle_mat


def read_joint_angles_from_goni(file_path, ref_time_vec):
    '''
    The dataset only includes the goni signals on the right leg
    '''
    goni_path = file_path.replace('ik', 'gon')
    '''
    The frequency of goni meter is higher than motion capture data, and here we downsample the goni signals.
    '''
    data = np.genfromtxt(goni_path, delimiter=',')[1:]
    time_vec = data[:, 0].reshape((-1, 1))
    ref_time_vec = ref_time_vec.reshape((1, -1))
    time_diff = np.abs(time_vec - ref_time_vec)  # (length of time vec * length of ref_time_vec)
    closest_indices = np.argmin(time_diff, axis=0)  # (length of ref_time_vec,)
    data = data[closest_indices]
    right_joint_angle_mat = data[:, [4, 3, 1]]
    return right_joint_angle_mat


def read_gait_phase(file_path=None):
    gait_phase_name_list = ['gcLeft', 'gcRight']
    heel_strike_phase_list = []
    toe_off_phase_list = []
    for gait_phase_name in gait_phase_name_list:
        gait_phase_path = file_path.replace('ik', gait_phase_name)
        data = np.array(np.genfromtxt(gait_phase_path, delimiter=',')[
                        1:])  # (gait phase based on heel strike, gait phase based on toe off, time)
        heel_strike_phase_list.append(data[:, 1])  # gait phase based on heel strike
        toe_off_phase_list.append(data[:, 2])
    heel_strike_phase_list.append(data[:, 0])
    toe_off_phase_list.append(data[:, 0])
    '''left gait phase, right gait phase, time'''
    heel_strike_phase_mat = np.stack(heel_strike_phase_list, axis=1)
    toe_off_phase_mat = np.stack(toe_off_phase_list, axis=1)
    return heel_strike_phase_mat, toe_off_phase_mat


def read_gait_mode(file_path=None):
    '''
    joint_angle_mat: [absolute angle of left hip in the sagital plane, absolute angle of right hip in the sagital plane]
    absolute angle means the angle is relative to the gravity direction.
    '''
    file_path = file_path.replace('ik', 'conditions')
    data = np.genfromtxt(file_path, delimiter=',', dtype=str)[1:]
    return data[:, 1]

def read_separate_gait_files(read_csv=True, data_file_path='data/separate_gait_data.npy'):
    if read_csv:
        folder_list = ['levelground', 'ramp', 'stair']
        condition_list = [['fast', 'normal', 'slow'],
                          np.arange(1, 7),
                          np.arange(1, 5)]
        joint_angle_and_velocity_mat_list = []
        gait_mode_list = []
        condition_mode_list = []
        foot_acc_mat_list = []
        for i in range(len(folder_list)):
            print(i)
            folder = folder_list[i]
            for j in range(len(condition_list[i])):
                print(j)
                condition = condition_list[i][j]
                data_dir = '../data/*/*/{}'.format(folder)
                file_path_list = glob.glob('{}/ik/*_{}_*.csv'.format(data_dir, condition))
                file_path_list = sorted(file_path_list)
                for file_path in file_path_list:
                    print(file_path)
                    
                    ##cxx add: read imu thigh acc for dataset_control
                    imu_file_path =file_path.replace('ik','imu')
                    foot_acc_mat = read_foot_acc(imu_file_path)
                    foot_acc_mat_list.append(foot_acc_mat[1:])
                    ##
                    
                    joint_angle_mat = read_joint_angles(file_path)
                    gait_mode_list.append(read_gait_mode(file_path)[1:])
                    heel_strike_phase_mat, toe_off_phase_mat = read_gait_phase(file_path)
                    '''The joint velocity of the first joint angle is not available, and thus we need to save data[1:].'''
                    joint_angle_and_velocity_mat_list.append(
                        Algo.calc_limit_cycle(joint_angle_mat, heel_strike_phase_mat))
                    condition_mode_list.append(j * np.ones(joint_angle_mat.shape[0]-1))
        gait_data = {'gait_mode': gait_mode_list,
                     'joint_angle_and_velocity': joint_angle_and_velocity_mat_list,
                     'condition': condition_mode_list,
                     ##cxx add
                     'foot_acc_mat':foot_acc_mat_list}
        np.save(data_file_path, gait_data)
    else:
        gait_data = np.load(data_file_path, allow_pickle=True).item()
    return gait_data



def read_joint_angles_with_gait_phases(read_csv=True, data_file_path='data/gait_data.npy'):
    if read_csv:
        folder_list = ['levelground', 'ramp', 'stair']
        condition_list = [['fast', 'normal', 'slow'],
                          np.arange(1, 7),
                          np.arange(1, 5)]
        joint_angle_and_velocity_mat_list = []
        gait_mode_list = []
        heel_strike_phase_list = []
        toe_off_phase_list = []
        condition_mode_list = []
        foot_acc_mat_list = []
        for i in range(len(folder_list)):
            print(i)
            folder = folder_list[i]
            for j in range(len(condition_list[i])):
                print(j)
                condition = condition_list[i][j]
                data_dir = '../data/*/*/{}'.format(folder)
                file_path_list = glob.glob('{}/ik/*_{}_*.csv'.format(data_dir, condition))
                file_path_list = sorted(file_path_list)
                for file_path in file_path_list:
                    print(file_path)
                    
                    ##cxx add: read imu thigh acc for dataset_control
                    imu_file_path =file_path.replace('ik','imu')
                    foot_acc_mat = read_foot_acc(imu_file_path)
                    foot_acc_mat_list.append(foot_acc_mat)
                    ##
                    
                    joint_angle_mat = read_joint_angles(file_path)
                    heel_strike_phase_mat, toe_off_phase_mat = read_gait_phase(file_path)
                    gait_mode_list.append(read_gait_mode(file_path)[1:])
                    '''The joint velocity of the first joint angle is not available, and thus we need to save data[1:].'''
                    joint_angle_and_velocity_mat_list.append(
                        Algo.calc_limit_cycle(joint_angle_mat, heel_strike_phase_mat))
                    heel_strike_phase_list.append(heel_strike_phase_mat[1:])
                    toe_off_phase_list.append(toe_off_phase_mat[1:])
                    condition_mode_list.append(j * np.ones(joint_angle_mat.shape[0]))
        joint_angle_and_velocity_mat = np.concatenate(joint_angle_and_velocity_mat_list, axis=0)
        gait_mode_vec = np.concatenate(gait_mode_list, axis=0)
        heel_strike_phase_mat = np.concatenate(heel_strike_phase_list, axis=0)
        toe_off_phase_mat = np.concatenate(toe_off_phase_list, axis=0)
        condition_vec = np.concatenate(condition_mode_list, axis=0)
        foot_acc_mat = np.concatenate(foot_acc_mat_list, axis=0)
        gait_data = {'heel_strike_phase_mat': heel_strike_phase_mat,
                     'toe_off_phase_mat': toe_off_phase_mat,
                     'gait_mode_vec': gait_mode_vec,
                     'joint_angle_and_velocity_mat': joint_angle_and_velocity_mat,
                     'condition_vec': condition_vec,
                     ##cxx add
                     'foot_acc_mat':foot_acc_mat}
        np.save(data_file_path, gait_data)
    else:
        gait_data = np.load(data_file_path, allow_pickle=True).item()
    return gait_data


def read_mean_joint_angle():
    '''
        joint_angle_mean_dict: {'stance', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, (gait_phase + joint number), types)
    '''
    phase_name_list = ['all']
    joint_angle_mean_dict = {}
    for phase_name in phase_name_list:
        joint_angle_mean_list = np.load('data/{}_joint_angle_mean.npy'.format(phase_name), allow_pickle=True)
        joint_angle_mean_dict[phase_name] = joint_angle_mean_list
    return joint_angle_mean_dict


def write_mean_and_mono_joint_angle():
    gait_data = read_joint_angles_with_gait_phases(read_csv=False)
    phase_name_list = ['all']
    for phase_name in phase_name_list:
        phase_joint_data, phase_condition = Algo.segment_gait_to_phases(gait_data, phase_name=phase_name)
        joint_angle_mean_list, joint_angle_mono_list = Algo.extract_monotonous_thigh_angle(phase_joint_data,
                                                                                           phase_condition,
                                                                                           phase_name=phase_name)
        np.save('data/{}_joint_angle_mean.npy'.format(phase_name), joint_angle_mean_list)
        np.save('data/{}_joint_angle_mono.npy'.format(phase_name), joint_angle_mono_list)


def write_mean_thigh_angle():
    gait_data = read_joint_angles_with_gait_phases(read_csv=False)
    left_and_right_joint_angle_list = []
    for leg_idx in range(2):
        phase_joint_data, phase_condition = Algo.segment_gait_to_phases(gait_data, phase_name='all', leg_idx=leg_idx,
                                                                        ref_leg_idx=leg_idx)
        joint_angle_mean_list = Algo.calc_mean_thigh_angle(phase_joint_data, phase_condition)
        left_and_right_joint_angle_list.append(joint_angle_mean_list)
    write_mean_thigh_angle_to_file(left_and_right_joint_angle_list)


def write_mean_thigh_angle_to_file(left_and_right_joint_angle_list):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 ms', '1.17 ms', '0.88 ms'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    for r in range(len(mode_list)):
        for c in range(len(legend_list[r])):
            left_joint_angle_mean =left_and_right_joint_angle_list[0][r][:, 0, c]
            right_joint_angle_mean =left_and_right_joint_angle_list[1][r][:, 0, c]
            joint_angle_mean = np.stack([left_joint_angle_mean, right_joint_angle_mean], axis=1)
            data = pd.DataFrame(joint_angle_mean, columns=['Left thigh angle', 'Right thigh angle'])
            data.to_csv('data/thigh_angle_mean_{}_{}.csv'.format(mode_list[r], legend_list[r][c]))


def write_thigh_angle():
    file_path_list = ['data/AB07/10_14_18/levelground/ik/levelground_ccw_normal_01_01.csv',
                      'data/AB07/10_14_18/stair/ik/stair_1_l_01_01.csv', ]
    joint_angle_mat_list = []
    for file_path in file_path_list:
        joint_angle_vec = read_joint_angles(file_path)
        heel_strike_phase_mat, toe_off_phase_mat = read_gait_phase(file_path)
        joint_angle_mat = np.concatenate([joint_angle_vec[:, [0, 3]], heel_strike_phase_mat], axis=-1)
        joint_angle_mat[:, -1] -= joint_angle_mat[0, -1]
        joint_angle_mat_list.append(joint_angle_mat)

    file_name_list = ['levelground', 'stair']
    for i in range(2):
        data = pd.DataFrame(joint_angle_mat_list[i],
                            columns=['Left thigh angle', 'Right thigh angle', 'Left gait cycle', 'Right gait cycle',
                                     'Times'])
        data.to_csv('data/{}.csv'.format(file_name_list[i]))


def read_image_to_video(img_name_vec, video_name, fps=2):
    if os.path.exists(video_name):
        os.remove(video_name)
    img_name_vec = sorted(img_name_vec)
    img_array = []
    for filename in img_name_vec:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # H.264 encoder
    out = cv2.VideoWriter(video_name, fourcc, fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def write_phase_joint_angle():
    gait_data = read_joint_angles_with_gait_phases(read_csv=False)
    for leg_idx in range(2):
        phase_joint_data, phase_condition = Algo.segment_gait_to_phases(gait_data, phase_name='all', leg_idx=leg_idx,
                                                                        ref_leg_idx=leg_idx)

