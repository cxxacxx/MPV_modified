import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
import time
import open3d as o3d
import scipy as sp
from Utils import IO, Plot
from matplotlib import cm
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.signal import butter, lfilter, freqz

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

default_cam_intri = np.asarray(
[[521.85359567,   0.        , 321.18647073],
[0.        , 521.7098714 , 233.81475134],
[0.        ,   0.        ,   1.        ]])


def read_cam_intrinsic():
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.width = 640
    pinhole_camera_intrinsic.height = 480
    pinhole_camera_intrinsic.intrinsic_matrix = default_cam_intri
    return pinhole_camera_intrinsic


def uv_vec2cloud(uv_vec, depth_img, depth_scale = 1e-3):
    '''
        uv_vec: n×2,
        depth_image: rows * cols
    '''
    fx = default_cam_intri[0, 0]
    fy = default_cam_intri[1, 1]
    cx = default_cam_intri[0, 2]
    cy = default_cam_intri[1, 2]
    cloud = np.zeros((len(uv_vec), 3))
    cloud[:, 2] = depth_img[uv_vec[:, 1].astype(int), uv_vec[:, 0].astype(int)] * depth_scale
    cloud[:, 0] = (uv_vec[:, 0] - cx) * (cloud[:, 2] / fx)
    cloud[:, 1] = (uv_vec[:, 1] - cy) * (cloud[:, 2] / fy)
    return cloud


def depth2cloud(img_depth, cam_intri_inv = None):
    # default unit of depth and point cloud is mm.
    if cam_intri_inv is None:
        cam_intri_inv = np.zeros((1, 1, 3, 3))
        cam_intri_inv[0, 0] = np.linalg.inv(default_cam_intri)
    uv_vec = np.transpose(np.mgrid[0:480, 0:640], (1, 2, 0))[:, :, [1, 0]].reshape((-1, 2))
    point_cloud = uv_vec2cloud(uv_vec, img_depth, depth_scale=1)
    return point_cloud


def filter_gait_event_indices(gait_event_indices, idx_diff_threshold = 30):
    idx_diff = 100 * np.ones(gait_event_indices.shape)
    '''
        If the distance between the current idx and the next idx is larger than a threshold, the current idx is valid.
    '''
    idx_diff[:-1] = gait_event_indices[1:] - gait_event_indices[:-1]
    filtered_indices = gait_event_indices[idx_diff > idx_diff_threshold]
    return filtered_indices


def fifo_data_vec(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat


def calc_leg_data(roll_vec):
    joint_angle_vec = calc_joint_angle(roll_vec)
    joint_x_vec, joint_y_vec = calc_joint_points(joint_angle_vec[-1])
    return joint_angle_vec, joint_x_vec, joint_y_vec


def calc_joint_angle(roll_vec, is_rad=True):
    if is_rad:
        roll_vec *= np.pi / 180
    return np.c_[roll_vec[:, 0], roll_vec[:, 1] - roll_vec[:, 0], roll_vec[:, 2] - roll_vec[:, 1]]


def calc_joint_points(joint_angle_vec):
    delta_x_vec = np.zeros(3)
    delta_y_vec = np.zeros(3)
    angle_bias = [0, 0, np.pi / 2]
    for i in range(3):
        delta_x_vec[i] = np.sin(np.sum(joint_angle_vec[:i + 1]) + angle_bias[i])
        delta_y_vec[i] = -np.cos(np.sum(joint_angle_vec[:i + 1]) + angle_bias[i])
    delta_x_vec[-1] *= 0.5
    delta_y_vec[-1] *= 0.5
    x_vec = np.zeros(4)
    y_vec = np.zeros(4)
    for i in range(1, 4):
        x_vec[i] = np.sum(delta_x_vec[:i])
        y_vec[i] = np.sum(delta_y_vec[:i])
    return x_vec, y_vec


def calc_limit_cycle(joint_angle_mat, heel_strike_phase_mat):
    joint_angle_velocity_mat = (joint_angle_mat[1:] - joint_angle_mat[:-1]) / (
            heel_strike_phase_mat[1:, [-1]] - heel_strike_phase_mat[:-1, [-1]])
    '''
    joint_angle_and_velocity_mat: (n * 12), 12 = joint angle + joint angular velocity
    joint angle: [left hip, knee, ankle, right hip knee, ankle]
    '''
    joint_angle_and_velocity_mat = np.concatenate([joint_angle_mat[1:], joint_angle_velocity_mat], axis=-1)
    return joint_angle_and_velocity_mat


def segment_trial(joint_angle_and_velocity_mat, gait_mode_vec, heel_strike_phase_mat, condition_vec):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    all_joint_data = np.zeros(len(mode_list), dtype=np.object)
    all_condition = np.zeros(len(mode_list), dtype=np.object)
    for m in range(len(mode_list)):
        mode = mode_list[m]
        is_current_mode_vec = (gait_mode_vec == mode).astype(np.int)
        mode_change_vec = is_current_mode_vec[1:] - is_current_mode_vec[:-1]
        '''Because the length of the mode_change_vec is n-1, we need to offset the indices'''
        start_indices = np.where(mode_change_vec == 1)[0] + 1
        end_indices = np.where(mode_change_vec == -1)[0] + 1
        joint_data_in_a_gait_list = []
        condition_list = []
        for i in range(len(start_indices)):
            indices = np.arange(start_indices[i], end_indices[i])
            for k in [1]:  # k = [0, 1], left and right leg, k = 0, only left, k = 1, only right
                gait_phase_i = heel_strike_phase_mat[indices, k]
                heel_strike_indices = np.where(gait_phase_i == 0)[0]
                if len(heel_strike_indices) > 1:
                    for j in range(len(heel_strike_indices) - 1):
                        gait_indices = np.arange(indices[heel_strike_indices[j]], indices[heel_strike_indices[j + 1]])
                        gait_phase_vec = heel_strike_phase_mat[gait_indices, k]
                        time_vec = heel_strike_phase_mat[gait_indices, [-1]]
                        '''Time, gait_phase, joint angle, joint velocity'''
                        joint_data_in_a_gait = np.c_[time_vec, gait_phase_vec.reshape((-1, 1)),
                                                     joint_angle_and_velocity_mat[gait_indices][:, [k, k + 2]]]
                        f = interpolate.interp1d(gait_phase_vec, joint_data_in_a_gait, kind='cubic', axis=0)
                        gait_phase_new = np.arange(0, 101)
                        joint_data_in_a_gait_new = f(gait_phase_new)
                        joint_data_in_a_gait_list.append(joint_data_in_a_gait_new)
                        condition_list.append(np.median(condition_vec[gait_indices]).astype(np.int))

        all_joint_data[m] = np.stack(joint_data_in_a_gait_list, axis=0)
        all_condition[m] = np.array(condition_list)
        for i in range(3):
            _, inlier_indices = remove_outliers(all_joint_data[m][:, :, 2 + i])
            all_joint_data[m] = all_joint_data[m][inlier_indices]
            all_condition[m] = all_condition[m][inlier_indices]
    return all_joint_data, all_condition


def segment_gait_to_phases(gait_data, phase_name='swing', leg_idx=0, ref_leg_idx=0, remove_outlier=True):
    joint_angle_and_velocity_mat, gait_mode_vec, heel_strike_phase_mat, toe_off_phase_mat, condition_vec = (
        gait_data['joint_angle_and_velocity_mat'], gait_data['gait_mode_vec'], gait_data['heel_strike_phase_mat'],
        gait_data['toe_off_phase_mat'], gait_data['condition_vec'])
    '''Segment a gait to stance and swing based on the heel strike event and toe off event.'''
    steady_mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    transition_mode_list = ['walk-stairascent', 'walk-stairdescent', 'walk-rampascent', 'walk-rampdescent',
                            'stairascent-walk', 'stairdescent-walk', 'rampascent-walk', 'rampdescent-walk', ]
    mode_list = steady_mode_list + transition_mode_list
    # all_joint_data = np.zeros(len(mode_list), dtype=np.object)
    # all_condition = np.zeros(len(mode_list), dtype=np.object)
    all_joint_data = {}
    all_condition = {}
    for m in range(len(mode_list)):
        mode = mode_list[m]
        is_current_mode_vec = (gait_mode_vec == mode).astype(np.int)
        mode_change_vec = is_current_mode_vec[1:] - is_current_mode_vec[:-1]
        '''Because the length of the mode_change_vec is n-1, we need to offset the indices'''
        start_indices = np.where(mode_change_vec == 1)[0] + 1
        end_indices = np.where(mode_change_vec == -1)[0] + 1
        joint_data_in_a_gait_list = []
        condition_list = []
        for i in range(len(start_indices)):
            '''The start and end of a mode.'''
            if mode in transition_mode_list:
                indices = np.arange(start_indices[i] - 40,
                                    end_indices[i])  # ensure the data include the last heel strike event
            else:
                indices = np.arange(start_indices[i], end_indices[i])
            # leg_idx = 0, only left, leg_idx = 1, only right
            heel_strike_phase_i = heel_strike_phase_mat[indices, ref_leg_idx]
            heel_strike_indices = np.where(heel_strike_phase_i == 0)[0]
            toe_off_phase_i = toe_off_phase_mat[indices, ref_leg_idx]
            toe_off_indices = np.where(toe_off_phase_i == 0)[0]
            if len(heel_strike_indices) > 1:
                for j in range(len(heel_strike_indices) - 1):
                    toe_off_idx = calc_toe_off_indices_between_two_heel_strike_events(
                        toe_off_indices, heel_strike_indices[j], heel_strike_indices[j + 1])
                    if toe_off_idx is None:
                        continue
                    if 'swing' in phase_name:
                        phase_indices = np.arange(indices[toe_off_idx], indices[heel_strike_indices[j + 1]])
                    elif 'stance' in phase_name:
                        phase_indices = np.arange(indices[heel_strike_indices[j]], indices[toe_off_idx])
                    else:
                        phase_indices = np.arange(indices[heel_strike_indices[j]],
                                                  indices[heel_strike_indices[j + 1]])
                    phase_vec = heel_strike_phase_mat[phase_indices, ref_leg_idx]
                    time_vec = heel_strike_phase_mat[phase_indices, [-1]]
                    scaled_phase_vec = (phase_vec - phase_vec[0]) * \
                                       101 / (phase_vec[-1] - phase_vec[0])
                    '''Time, gait_phase, joint angle, joint velocity'''
                    joint_angle_leg_k = joint_angle_and_velocity_mat[phase_indices][:,
                                        (3 * leg_idx):(3 * (leg_idx + 1))]
                    joint_velocity_leg_k = joint_angle_and_velocity_mat[phase_indices][:,
                                           (6 + (3 * leg_idx)):(6 + (3 * (leg_idx + 1)))]
                    joint_data_in_a_gait = np.c_[time_vec, scaled_phase_vec.reshape((-1, 1)),
                                                 joint_angle_leg_k, joint_velocity_leg_k]
                    f = interpolate.interp1d(scaled_phase_vec, joint_data_in_a_gait, kind='cubic', axis=0)
                    gait_phase_new = np.arange(0, 101)
                    joint_data_in_a_gait_new = f(gait_phase_new)
                    joint_data_in_a_gait_list.append(joint_data_in_a_gait_new)
                    condition_list.append(np.median(condition_vec[phase_indices]).astype(np.int))
        if len(joint_data_in_a_gait_list) > 0:
            all_joint_data[mode_list[m]] = np.stack(joint_data_in_a_gait_list, axis=0)
            all_condition[mode_list[m]] = np.array(condition_list)
            '''Remove outlier according to the hip angle'''
            if remove_outlier:
                _, inlier_indices = remove_outliers(all_joint_data[mode_list[m]][:, :, 2])
                all_joint_data[mode_list[m]] = all_joint_data[mode_list[m]][inlier_indices]
                all_condition[mode_list[m]] = all_condition[mode_list[m]][inlier_indices]
        else:
            all_joint_data[mode_list[m]] = None
            all_condition[mode_list[m]] = None
    return all_joint_data, all_condition


def calc_monotonous_joint_angle_indices(joint_angle_vec, phase_name):
    '''
        phase_joint_data: (phase_cycle, )
    '''
    if 'swing' in phase_name:
        stop_idx = np.argmax(joint_angle_vec[30:]) + 30
        joint_angle_diff = joint_angle_vec[1:stop_idx] - joint_angle_vec[:stop_idx - 1]
        negative_gradient_indices = np.where(joint_angle_diff < 0)[0]
        if len(negative_gradient_indices) > 0:
            stop_idx = negative_gradient_indices[0]
        indices = np.arange(start=0, stop=stop_idx)
    else:
        stop_idx = np.argmin(joint_angle_vec[30:]) + 30
        joint_angle_diff = joint_angle_vec[1:stop_idx] - joint_angle_vec[:stop_idx - 1]
        positive_gradient_indices = np.where(joint_angle_diff > 0)[0]
        if len(positive_gradient_indices) > 0:
            start_idx = positive_gradient_indices[-1] + 2
        else:
            start_idx = 1
        indices = np.arange(start=start_idx, stop=np.argmin(joint_angle_vec[30:]) + 30)
    return indices


def fit_joint_angle_mat(indices, joint_angle_mat):
    scaled_indices = (indices - indices[0]) * 101 / (indices[-1] - indices[0])
    '''Time, gait_phase, joint angle, joint velocity'''
    f = interpolate.interp1d(scaled_indices, joint_angle_mat, kind='cubic', axis=0)
    indices_new = np.arange(0, 101)
    joint_angle_mat = f(indices_new)
    return joint_angle_mat


def calc_toe_off_indices_between_two_heel_strike_events(toe_off_indices_vec, last_heel_strike, current_heel_strike):
    valid_toe_off_indices = toe_off_indices_vec[np.logical_and(toe_off_indices_vec > last_heel_strike,
                                                               toe_off_indices_vec < current_heel_strike)]
    if len(valid_toe_off_indices) > 0:
        return valid_toe_off_indices[0]
    else:
        return None


def calc_angle_integration(all_joint_data, all_condition, offset_type='gait_mean'):
    for i in range(len(all_joint_data)):
        joint_data = all_joint_data[i]
        joint_angle_mat = joint_data[..., 2]
        if 'gait' in offset_type:
            # mean_angle = np.mean(joint_angle_mat, axis=-1, keepdims=True)
            # mean_angle[1:] = mean_angle[:-1]
            # joint_angle_mat -= mean_angle
            joint_angle_mat -= np.mean(joint_angle_mat, axis=-1, keepdims=True)
        elif 'global' in offset_type:
            joint_angle_mat -= np.mean(joint_angle_mat)
        else:
            for j in range(len(np.unique(all_condition[i]))):
                indices = all_condition[i] == j
                joint_angle_mat[indices] -= np.mean(joint_angle_mat[indices])
        joint_angle_integration_mat = np.zeros(joint_angle_mat.shape)
        time_mat = joint_data[..., 1]
        dt_mat = time_mat[:, 1:] - time_mat[:, :-1]
        delta_angle_mat = joint_angle_mat[:, :-1] * dt_mat
        n = delta_angle_mat.shape[-1]
        A = np.tril(np.ones((n, n)), 0).reshape((1, n, n))
        delta_angle_mat = delta_angle_mat.reshape((-1, n, 1))
        joint_angle_integration_mat[:, 1:] = np.matmul(A, delta_angle_mat).squeeze()
        '''Time, gait_phase, joint angle, joint velocity, joint angle integration'''
        joint_data_new = np.zeros((joint_data.shape[0], joint_data.shape[1], 5))
        joint_data_new[..., :4] = joint_data[..., :4]
        joint_data_new[..., -1] = joint_angle_integration_mat
        all_joint_data[i] = joint_data_new

    return all_joint_data


def calc_angle_integration_vec(joint_vec, dt=1e-2):
    delta_angle_mat = joint_vec[:-1] * dt
    n = delta_angle_mat.shape[-1]
    A = np.tril(np.ones((n, n)), 0)
    delta_angle_mat = delta_angle_mat
    joint_integration_vec = np.zeros(joint_vec.shape)
    joint_integration_vec[1:] = np.matmul(A, delta_angle_mat).squeeze()
    return joint_integration_vec


def calc_phase_variable(all_joint_data, is_filter=True):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    all_gait_phase = np.zeros(len(mode_list), dtype=np.object)
    '''correlation_coefficient_mat = [coefficient between the actual phase and the phase predicted by q_int_q,
                                      coefficient between the actual phase and the phase predicted by q_d_q,]'''
    correlation_coefficient_mat = np.zeros((len(mode_list), 2))
    rmse_mat = np.zeros((len(mode_list), 2))
    for i in range(len(mode_list)):
        joint_data = all_joint_data[i]
        '''gait phase = [actual phase, phase predicted by q_int_q, phase predicted by q_d_q]'''
        gait_phase = np.zeros((joint_data.shape[0], joint_data.shape[1], 3))
        print(joint_data.shape)
        gait_phase[..., 0] = joint_data[..., 1]
        time_mat = joint_data[..., 0]
        # don't know the value in real-time computing?
        gamma = (np.max(joint_data[..., 2:], axis=1, keepdims=True) + np.min(joint_data[..., 2:], axis=1,
                                                                             keepdims=True)) / 2
        ratio = (np.max(joint_data[..., 2:], axis=1, keepdims=True) - np.min(joint_data[..., 2:], axis=1,
                                                                             keepdims=True))

        joint_data[..., 2:] -= gamma
        gait_phase[..., 1] = np.arctan2(joint_data[..., 4] * (ratio[..., 0] / ratio[..., 2]),
                                        joint_data[..., 2])  # q_int_q
        gait_phase[..., 2] = np.arctan2(joint_data[..., 3] * (ratio[..., 0] / ratio[..., 1]),
                                        joint_data[..., 2])  # q_d_q
        gait_phase[..., 1:] = 100 * ((gait_phase[..., 1:] - gait_phase[:, [0], 1:]) / (2 * np.pi))
        gait_phase[..., 2] *= -1
        correlation_coefficient_vec = np.zeros((gait_phase.shape[0], 2))
        rmse_vec = np.zeros((gait_phase.shape[0], 2))
        for r in range(gait_phase.shape[0]):
            for c in range(2):
                gait_phase[r, :, c + 1] = correct_measurement(gait_phase[r, :, c + 1], period=100)
                if is_filter:
                    gait_phase[r, :, c + 1] = filter_gait_phase(gait_phase[r, :, c + 1],
                                                                dt=time_mat[r, 1] - time_mat[r, 0])
                correlation_coefficient_vec[r, c], _ = pearsonr(gait_phase[r, :, 0], gait_phase[r, :, c + 1])
                rmse_vec[r, c] = np.sqrt(mse(gait_phase[r, :, 0], gait_phase[r, :, c + 1]))
        correlation_coefficient_mat[i] = np.mean(correlation_coefficient_vec, axis=0)
        rmse_mat[i] = np.mean(rmse_vec, axis=0)
        all_gait_phase[i] = gait_phase
        # print(gait_phase)
    print(correlation_coefficient_mat)
    print(rmse_mat)
    return all_gait_phase, correlation_coefficient_mat, rmse_mat


def remove_outliers_of_joint_angle_mat(joint_angle_mat, label_vec, std_ratio=3):
    joint_angle_mat_list = []
    label_vec_list = []
    for i in range(5):
        joint_angle_mat_i = joint_angle_mat[label_vec == i]
        _, inlier_indices = remove_outliers(joint_angle_mat_i, std_ratio=std_ratio)
        joint_angle_mat_i = joint_angle_mat_i[inlier_indices]
        label_vec_list.append(label_vec[label_vec == i][inlier_indices].reshape(-1))
        joint_angle_mat_list.append(joint_angle_mat_i)
    joint_angle_mat = np.concatenate(joint_angle_mat_list, axis=0)
    label_vec = np.concatenate(label_vec_list, axis=0)
    return joint_angle_mat, label_vec

def remove_outliers(joint_angle_mat, std_ratio=1):
    '''
    joint angle mat: trial number * gait cycle length (101)
    Calculate the trail for which the curve is statistically different from the mean of curves.
    std_ratio, ratio of reserved data:
    std = 1, 68.3%
    std = 2, 95.5%
    std = 3, 99.7%
    '''
    joint_angle_mat = joint_angle_mat.reshape((joint_angle_mat.shape[0], -1))
    inlier_indices = np.arange(len(joint_angle_mat))
    joint_angle_mean = np.mean(joint_angle_mat[inlier_indices], axis=0, keepdims=True)
    rmse_vec = rmse(joint_angle_mat, joint_angle_mean, axis=-1)
    rmse_mean = np.mean(rmse_vec)
    rmse_vec -= rmse_mean
    inlier_indices = rmse_vec < std_ratio * np.std(rmse_vec)
    return joint_angle_mat[inlier_indices], inlier_indices


def calc_mean_thigh_angle(all_joint_data, all_condition):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    joint_angle_mean_list = np.zeros(len(mode_list), dtype=np.object)
    for i in range(len(mode_list)):
        joint_angle_mean_vec = np.zeros((101, 3, len(legend_list[i])))
        for j in range(len(np.unique(all_condition[mode_list[i]]))):
            joint_angle_mat = all_joint_data[mode_list[i]][all_condition[mode_list[i]] == j]
            joint_angle_mat = joint_angle_mat[..., 2:5]  # thigh, knee, ankle
            if joint_angle_mat.shape[0] > 1:
                joint_angle_mat = np.mean(joint_angle_mat, axis=0)
            elif joint_angle_mat.shape[0] == 1:
                joint_angle_mat = joint_angle_mat[0]
            else:
                joint_angle_mat = np.zeros((101, 3))
                print('!!!!!!!{}, {}, Error of joint angle!!!!!!!!!!'.format(mode_list[i], legend_list[i][j]))
            joint_angle_mean_vec[:, :, j] = joint_angle_mat  # hip, knee, angkle
        joint_angle_mean_list[i] = joint_angle_mean_vec
    return joint_angle_mean_list


def extract_monotonous_thigh_angle(all_joint_data, all_condition, phase_name='stance'):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    joint_angle_mean_list = np.zeros(len(mode_list), dtype=np.object)
    joint_angle_mono_list = np.zeros(len(mode_list), dtype=np.object)
    for i in range(len(mode_list)):
        joint_angle_mean_vec = np.zeros((101, 3, len(legend_list[i])))
        joint_angle_mono_vec = np.zeros((101, 3, len(legend_list[i])))
        for j in range(len(np.unique(all_condition[mode_list[i]]))):
            joint_angle_mat = all_joint_data[mode_list[i]][all_condition[mode_list[i]] == j]
            joint_angle_mat = joint_angle_mat[..., 2:5]  # thigh, knee, ankle
            if joint_angle_mat.shape[0] > 1:
                joint_angle_mat = np.mean(joint_angle_mat, axis=0)
            elif joint_angle_mat.shape[0] == 1:
                joint_angle_mat = joint_angle_mat[0]
            else:
                joint_angle_mat = None
                print('!!!!!!!Error of joint angle!!!!!!!!!!')
            indices = calc_monotonous_joint_angle_indices(joint_angle_mat[:, 0], phase_name)
            joint_angle_mono_mat = fit_joint_angle_mat(indices, joint_angle_mat[indices])
            joint_angle_mean_vec[:, :, j] = joint_angle_mat  # hip, knee, angkle
            joint_angle_mono_vec[:, :, j] = joint_angle_mono_mat  # hip, knee, angkle
        joint_angle_mean_list[i] = joint_angle_mean_vec
        joint_angle_mono_list[i] = joint_angle_mean_vec
    return joint_angle_mean_list, joint_angle_mono_list


def correct_measurement(val_vec, period=1):
    for i in range(1, val_vec.shape[0]):
        val = val_vec[i]
        val_list = np.array([val - period, val, val + period])
        dist_vec = np.abs(val_list - val_vec[i - 1])
        ''' select the value that is closest to the last value '''
        val_vec[i] = val_list[np.argmin(dist_vec)]
    return val_vec


def estimate_state_velocity(measurements, dt):
    measurements_new = np.zeros((measurements.shape[0], 2))
    measurements_new[:, 0] = measurements.squeeze()
    measurements_new[0, 1] = 1
    for i in range(1, measurements.shape[0]):
        measurements_new[i, 1] = (measurements[i] - measurements[i-1]) / dt
    return measurements_new


def filter_measurements(measurements, P, Q, R, F, H, v_range = [0, 100]):
    dt = F[0, -1]
    measurements = estimate_state_velocity(measurements, dt)
    # measurements[:, 1] = np.clip(measurements[:, 1], a_min=v_range[0], a_max=v_range[1])
    kf = KalmanFilter(P=P, Q=Q, R=R, F=F, H=H, x_0=measurements[0])
    filter_result = np.zeros(measurements.shape)
    filter_result[0] = measurements[0]
    for i in range(1, measurements.shape[0]):
        filter_result[i] = kf.forward(measurements[i], F)
    return filter_result


def mse(A, B, axis=-1):
    return np.mean((A - B) ** 2, axis=axis)


def rmse(A, B, axis=-1):
    return np.sqrt(mse(A, B, axis=axis))


def filter_gait_phase(gait_phase, dt):
    R = np.array([[0.5, 0],
                  [0, 5]])  # measurement noise covariance
    Q = np.array([[0, 0],
                  [0, 1e-4]])  # process noise covariance
    P = 0.1 * R  # uncertainty covariance
    F = np.array([[1, dt],
                  [0, 1]])
    H = np.array([[1., 0.],
                  [0., 1.]])
    filter_phase_mat = filter_measurements(gait_phase, P, Q, R, F, H)
    return filter_phase_mat[:, 0]


def analyze_phase_variable():
    gait_data = IO.read_joint_angles_with_gait_phases(read_csv=False)
    # offset_type_list = ['gait_mean', 'global_mean', 'mode_mean']
    offset_type_list = ['gait_mean']
    phase_name_list = ['stance', 'swing']
    for phase_name in phase_name_list:
        phase_joint_data, phase_condition = segment_gait_to_phases(gait_data, phase_name=phase_name)
        for plot_mean in [True]:
            Plot.plot_thigh_angle(phase_joint_data, phase_condition, phase_name='{}'.format(phase_name),
                                  plot_mean=plot_mean)
        plt.show()
    # plot_initial_mean_angle(all_joint_data, all_condition, image_name='initial_mean_angle')
    # for offset_type in offset_type_list:
    #     all_joint_data, all_condition = segment_trial(gait_data['joint_angle_and_velocity_mat'],
    #                                                   gait_data['gait_mode_vec'],
    #                                                   gait_data['heel_strike_phase_mat'], gait_data['condition_vec'])
    #     all_joint_data = calc_angle_integration(all_joint_data, all_condition, offset_type)
    #     # plot_q_int_q(all_joint_data, image_name='q_int_q')
    #     # plot_normalized_limit_cycle(all_joint_data, image_name='q_dq')
    #     plot_phase_cycle(all_joint_data, all_condition=None, image_name='phase_cycle_offset_{}'.format(offset_type))
    #     all_gait_phase = np.zeros(2, dtype=np.object)
    #     correlation_coefficient_mat = np.zeros(2, dtype=np.object)
    #     rmse_mat = np.zeros(2, dtype=np.object)
    #     is_filter_list = [True, False]
    #     for i in range(2):
    #         all_gait_phase[i], correlation_coefficient_mat[i], rmse_mat[i] = calc_phase_variable(
    #             all_joint_data, is_filter=is_filter_list[i])
    #     plot_gait_phase(all_gait_phase, correlation_coefficient_mat, rmse_mat,
    #                     image_name='gait_phase_offset_{}'.format(offset_type), method_idx=0)


def fun_x_to_y(x, h, x0, k, b):
    y = h / (1 + np.exp(-k * (x - x0))) + b  # sigmoid
    # y = h * np.tan(k * (x - x0)) + b # tan
    return (y)


def fun_y_to_x(y, h, x0, k, b):
    x = x0 - np.log(h / (y - b) - 1) / k  # sigmoid
    # x = x0 + np.arctan((y-b)/h)/k # tan
    return (x)


def fit_phase():
    phase_name_list = ['stance', 'swing']
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(phase_name_list)):
        phase_name = phase_name_list[i]
        joint_angle_list = np.load('data/{}_joint_angle.npy'.format(phase_name),
                                   allow_pickle=True)
        phase_paras_list = np.zeros(len(joint_angle_list), dtype=np.object)
        cm_fun = cm.get_cmap('tab10', 10)
        for r in range(len(joint_angle_list)):
            phase_paras_mat = np.zeros((4, len(legend_list[r])))
            plt.subplot(2, len(joint_angle_list), i * len(joint_angle_list) + r + 1)
            for c in range(joint_angle_list[r].shape[-1]):
                y = joint_angle_list[r][:, 0, c]  # (time steps, joints, model number)
                x = np.arange(101)
                if np.argmax(y) > np.argmin(y):
                    popt = [np.max(y) - np.min(y), np.median(x), 1, np.min(y)]  # h, x0, k, b
                else:
                    popt = [np.max(y) - np.min(y), np.median(x), -1, np.min(y)]  # h, x0, k, b
                popt, pcov = curve_fit(fun_x_to_y, x, y, popt, method='lm')
                phase_paras_mat[:, c] = np.array(popt)
                y_hat = fun_x_to_y(x, *popt)
                plt.plot(x, y, '.', color=cm_fun(c), markersize=1)
                plt.plot(x, y_hat, color=cm_fun(c))
            plt.legend(legend_list[r], frameon=False, handlelength=0.5)
            plt.xlabel('{} phase (%)\n{}'.format(phase_name, mode_list[r]))
            if r == 0:
                plt.ylabel(r'Thigh angle (deg)')
            phase_paras_list[r] = phase_paras_mat
        np.save('data/{}_phase_paras.npy'.format(phase_name), phase_paras_list)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'fitted_thigh_angle'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def calc_phase():
    phase_name_list = ['stance', 'swing']
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    fig = plt.figure(figsize=(16, 10))
    for i in range(len(phase_name_list)):
        phase_name = phase_name_list[i]
        joint_angle_list = np.load('data/{}_joint_angle.npy'.format(phase_name),
                                   allow_pickle=True)
        phase_paras_list = np.load('data/{}_phase_paras.npy'.format(phase_name),
                                   allow_pickle=True)
        cm_fun = cm.get_cmap('tab10', 10)
        for r in range(len(joint_angle_list)):
            plt.subplot(2, len(joint_angle_list), i * len(joint_angle_list) + r + 1)
            for c in range(joint_angle_list[r].shape[-1]):
                y = joint_angle_list[r][:, 0, c]
                x = np.arange(101)
                phase_paras = phase_paras_list[r][:, c]
                y_1, y_2 = (fun_x_to_y(0, *phase_paras), fun_x_to_y(100, *phase_paras))
                # if 'stairdescent' == mode_list[r] and 'stance' in phase_name:
                #     y[:30] = y_1
                y = np.clip(y, min(y_1, y_2), max(y_1, y_2))
                x_hat = fun_y_to_x(y, *phase_paras)
                x_hat[0] = 0
                x_hat_filter = filter_gait_phase(x_hat, dt=1)
                r_corr, _ = pearsonr(np.arange(101), x_hat)
                rmse = np.sqrt(mse(np.arange(101), x_hat))
                print('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
                plt.plot(x, x_hat, '--', linewidth=1, color=cm_fun(c))
                plt.plot(x, x_hat_filter, color=cm_fun(c))
                plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            plt.legend(legend_list[r], frameon=False, handlelength=0.5)
            plt.xlabel('{} phase (%)\n{}'.format(phase_name, mode_list[r]))
            if r == 0:
                plt.ylabel(r'Predicted phase')
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'fitted_phase'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def calc_phase_point(theta_vec, phase_paras, x_low=0, x_high=100):
    theta_low, theta_high = (fun_x_to_y(x_low, *phase_paras), fun_x_to_y(x_high, *phase_paras))
    theta_vec = np.clip(theta_vec, min(theta_low, theta_high), max(theta_low, theta_high))
    phase_predicted = fun_y_to_x(theta_vec, *phase_paras)
    phase_predicted = np.clip(phase_predicted, x_low, x_high)
    return phase_predicted




def estimate_all_phase(joint_angle_mean_dict):
    '''
        joint_angle_mean_dict: {'all', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, joint number, types)
    '''
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    phase_name = 'all'
    all_phase_list = np.zeros(5, dtype=np.object)
    for r in range(len(mode_list)):
        all_phase_mat = np.zeros((101, len(legend_list[r])))
        for c in range(len(legend_list[r])):
            thigh_angle_vec = joint_angle_mean_dict[phase_name][r][:, 0, c]
            phase_vec = estimate_phase_from_thigh_angle(thigh_angle_vec)#, mode_list[r])
            all_phase_mat[:, c] = phase_vec
        all_phase_list[r] = all_phase_mat
    ##cxx debug
    # r = 2
    # c = 3
    # all_phase_mat = np.zeros((101, len(legend_list[r])))
    # thigh_angle_vec = joint_angle_mean_dict[phase_name][r][:, 0, c]
    # phase_vec = estimate_phase_from_thigh_angle(thigh_angle_vec)#, mode_list[r])
    # all_phase_mat[:, c] = phase_vec
    # all_phase_list[r] = all_phase_mat
    return all_phase_list


def calc_phase_parameters(thigh_angle_mean):
    stance_end_idx = np.argmin(thigh_angle_mean[10:70])+10
    stance_end_threshold = thigh_angle_mean[stance_end_idx]
    
    swing_end_threshold = np.max(thigh_angle_mean[30:90])
    swing_end_idx = np.argmax(thigh_angle_mean[30:90]) + 30
    idx_list = [0, stance_end_idx, swing_end_idx, 101]
    popt_list = []
    for r in range(len(idx_list) - 1):
        x = np.arange(idx_list[r], idx_list[r + 1])
        popt = fit_phase_of_monotonous_vec(x, thigh_angle_mean[x])
        popt_list.append(popt)
    paras_dict = {'stance_end_threshold': stance_end_threshold,
                  'swing_end_threshold': swing_end_threshold,
                  'popt_list': popt_list,
                  'idx_list': idx_list}
    return paras_dict


def estimate_phase_from_thigh_angle(thigh_angle_vec):
    plt.plot(thigh_angle_vec)
    stance_end_idx = np.argmin(thigh_angle_vec[10:60])+10
    init_idx = 30
    kickback_threshold = np.max(thigh_angle_vec[init_idx:]) - 0.5
    swing_end_idx = np.where(thigh_angle_vec[init_idx:] > kickback_threshold)[0][0] + init_idx
    if swing_end_idx < 90:
        idx_list = [0, stance_end_idx, swing_end_idx, 101]
    else:
        idx_list = [0, stance_end_idx, 101]
    phase_list = []
    for r in range(len(idx_list) - 1):
        x = np.arange(idx_list[r], idx_list[r + 1])
        popt = fit_phase_of_monotonous_vec(x, thigh_angle_vec[x])
        x_hat = calc_phase_point(thigh_angle_vec[x], popt, x_low=x[0], x_high=x[-1])
        phase_list.append(x_hat)
    phase_vec = np.concatenate(phase_list, axis=0)
    phase_vec[:5] = np.arange(5)
    phase_vec = filter_gait_phase(phase_vec, dt=1)
    phase_vec = np.clip(phase_vec, a_min=0, a_max=100)
    return phase_vec


def fit_phase_of_monotonous_vec(x, y):
    idx_max, idx_min = (np.argmax(y), np.argmin(y))
    k = 5 / (x[idx_max] - x[idx_min])
    x0 = np.median(x)
    h = (y[idx_max] - y[idx_min]) / (
                1 / (1 + np.exp(-k * (x[idx_max] - x0))) - 1 / (1 + np.exp(-k * (x[idx_min] - x0))))
    b = y[idx_max] - h
    pint = [h, x0, k, b]
    try:
        popt, pcov = curve_fit(fun_x_to_y, x, y, pint, method='trf', max_nfev=1e6)
    except:
        print('Achieved maximum iterations and used the initial guess as the optimal values of the fitted function.')
        popt = pint
    return popt


def estimate_all_angle(joint_angle_mean_dict, all_phase_list):
    '''
        joint_angle_mean_dict: {'stance', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, joint number, types)
        all_phase_list: [mode number]
        all_phase_list[0]: (200, types)
    '''
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    predicted_joint_angle_list = np.zeros(len(mode_list), dtype=np.object)
    joint_angle_mean_list = joint_angle_mean_dict['all']
    for r in range(len(mode_list)):
        predicted_joint_angle_mat = np.zeros((101, 3, len(legend_list[r])))
        for c in range(len(legend_list[r])):
            for j in range(1, 3):
                joint_angle_vec = joint_angle_mean_list[r][:, j, c]
                ideal_phase_vec = np.arange(101)
                f = interpolate.interp1d(ideal_phase_vec, joint_angle_vec, kind='cubic', axis=0)
                predicted_phase = all_phase_list[r][:, c]
                predicted_phase = np.clip(predicted_phase, np.min(ideal_phase_vec), np.max(ideal_phase_vec))
                predicted_joint_angle_vec = f(predicted_phase)
                predicted_joint_angle_mat[:, j, c] = predicted_joint_angle_vec
        predicted_joint_angle_list[r] = predicted_joint_angle_mat
    return predicted_joint_angle_list


def estimate_stance_swing_phase(joint_angle_mean_dict):
    '''
        joint_angle_mean_dict: {'stance', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, joint number, types)
    '''
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    phase_name_list = ['stance', 'swing']
    all_phase_list = np.zeros(5, dtype=np.object)
    for r in range(len(mode_list)):
        all_phase_mat = np.zeros((200, len(legend_list[r])))
        for c in range(len(legend_list[r])):
            all_phase_vec = np.zeros(200)
            for i in range(len(phase_name_list)):
                phase_name = phase_name_list[i]
                phase_paras_list = np.load('data/{}_phase_paras.npy'.format(phase_name),
                                           allow_pickle=True)
                phase_paras = phase_paras_list[r][:, c]
                thigh_angle_vec = joint_angle_mean_dict[phase_name][r][:, 0, c]
                phase_vec = np.arange(101)
                indices = calc_monotonous_joint_angle_indices(thigh_angle_vec, phase_name)
                phase_vec[:indices[0]] = 100 * i
                phase_vec[indices[-1] + 1:] = 100 * (i + 1)
                phase_vec[indices] = calc_phase_point(thigh_angle_vec[indices], phase_paras, x_low=100 * i)
                all_phase_vec[100 * i:100 * (i + 1)] = phase_vec[:100]
            all_phase_mat[:, c] = all_phase_vec
        all_phase_list[r] = all_phase_mat
    return all_phase_list


def estimate_stance_swing_angle(joint_angle_mean_dict, all_phase_list):
    '''
        joint_angle_mean_dict: {'stance', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, joint number, types)
        all_phase_list: [mode number]
        all_phase_list[0]: (200, types)
    '''
    phase_name_list = ['stance', 'swing']
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    predicted_joint_angle_list = np.zeros(len(mode_list), dtype=np.object)
    for r in range(len(mode_list)):
        predicted_joint_angle_mat = np.zeros((200, 3, len(legend_list[r])))
        for c in range(len(legend_list[r])):
            for j in range(1, 3):
                predicted_joint_angle_vec = np.zeros(200)
                for i in range(len(phase_name_list)):
                    phase_name = phase_name_list[i]
                    joint_angle_vec = joint_angle_mean_dict[phase_name][r][:, j, c]
                    ideal_phase_vec = np.arange(100 * i, 100 * (i + 1) + 1)
                    f = interpolate.interp1d(ideal_phase_vec, joint_angle_vec, kind='cubic', axis=0)
                    predicted_phase = all_phase_list[r][:, c][100 * i:100 * (i + 1)]
                    predicted_phase = np.clip(predicted_phase, np.min(ideal_phase_vec), np.max(ideal_phase_vec))
                    predicted_joint_angle_vec[100 * i:100 * (i + 1)] = f(predicted_phase)
                predicted_joint_angle_mat[:, j, c] = predicted_joint_angle_vec
        predicted_joint_angle_list[r] = predicted_joint_angle_mat
    return predicted_joint_angle_list


def segment_phase_based_on_thigh_angle(joint_angle_mean_dict):
    '''
        input joint_angle_mean_dict: {'stance', 'swing'}
        output joint_angle_mean_dict: {'stance', 'swing', 'kickback'}
    '''
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    phase_name_list = ['swing', 'kickback']
    joint_angle_mean_dict[phase_name_list[1]] = np.zeros(len(mode_list), dtype=np.object)
    for r in range(len(mode_list)):
        joint_angle_mean_dict[phase_name_list[1]][r] = np.zeros((101, 4, len(legend_list[r])))
        for c in range(len(legend_list[r])):
            joint_angle_mat = np.copy(joint_angle_mean_dict[phase_name_list[0]][r][:, :, c])
            thigh_angle_vec = joint_angle_mat[:, 0]
            kickback_threshold = np.max(thigh_angle_vec) - 2
            kickback_idx = np.where(thigh_angle_vec > kickback_threshold)[0][0] + 1
            sub_phase_indices_list = [np.arange(0, kickback_idx),
                                      np.arange(kickback_idx, len(thigh_angle_vec))]
            for i in range(len(phase_name_list)):
                sub_phase_indices = sub_phase_indices_list[i]
                joint_angle_mean_dict[phase_name_list[i]][r][:, :, c] = interpolate_joint_angle(
                    joint_angle_mat[sub_phase_indices], sub_phase_indices)
    return joint_angle_mean_dict


def interpolate_joint_angle(joint_angle_mat, indices, number=101):
    indices = (number - 1) * (indices - indices[0]) / (indices[-1] - indices[0])
    f = interpolate.interp1d(indices, joint_angle_mat, kind='cubic', axis=0)
    indices_new = np.arange(0, number)
    joint_angle_mat_new = f(indices_new)
    return joint_angle_mat_new


def simulate_thigh_angle(phase):
    phi = 2 * np.pi * 1e-2 * phase  # [0, np.pi]
    thigh_angle = 25 * np.cos(phi) + 25
    return thigh_angle


def calc_phase_based_on_q(thigh_angle_vec):
    '''Fit the phase based on thigh angle'''
    popt_list = []
    for i in range(2):
        x = np.arange(50 * i, 50 * (i + 1))
        ideal_thigh_angle_vec = simulate_thigh_angle(x)
        popt = fit_phase_of_monotonous_vec(x, ideal_thigh_angle_vec)
        popt_list.append(popt)

    phase_type = 0
    phase_vec = np.zeros(thigh_angle_vec.shape)
    for i in range(len(thigh_angle_vec)):
        if 0 == phase_type and thigh_angle_vec[i] < np.min(thigh_angle_vec) + 1e-2:
            phase_type = 1
        if 1 == phase_type and thigh_angle_vec[i] > np.max(thigh_angle_vec) - 1e-2:
            phase_type = 0
        phase_vec[i] = calc_phase_point(thigh_angle_vec[i], popt_list[phase_type],
                                        x_low=50 * phase_type, x_high=50 * (phase_type + 1))
    x_vec = phase_vec
    y_vec = thigh_angle_vec
    prev = 0
    for i in range(1, len(phase_vec)):
        if phase_vec[i] < phase_vec[i-1] - 50:
            print(phase_vec[i])
            if prev < i:
                phase_vec[prev:i] = filter_gait_phase(phase_vec[prev:i], dt = 1)
            prev = i
    phase_vec = np.clip(phase_vec, 0, 100)
    return phase_vec, x_vec, y_vec


def calc_phase_based_on_q_int_q(thigh_angle_vec):
    mean_angle = np.mean(thigh_angle_vec[0:100])
    dt = 1e-2
    joint_integration_vec = calc_angle_integration_vec(thigh_angle_vec[0:100] - mean_angle, dt=dt)
    q_int_max, q_int_min = (np.max(joint_integration_vec), np.min(joint_integration_vec))
    q_max, q_min = (np.max(thigh_angle_vec[0:100]), np.min(thigh_angle_vec[0:100]))
    mean_x = 0.5 * (q_max + q_min)
    mean_y = 0.5 * (q_int_max + q_int_min)
    x_y_ratio = (q_max - q_min) / (q_int_max - q_int_min)
    x_vec = np.zeros(thigh_angle_vec.shape)
    y_vec = np.zeros(thigh_angle_vec.shape)
    phase_vec = np.zeros(thigh_angle_vec.shape)
    last_heel_strike_idx = 0
    for i in range(len(thigh_angle_vec)):
        x_vec[i] = thigh_angle_vec[i]
        if thigh_angle_vec[i] > np.max(thigh_angle_vec) - 1e-2:
            last_heel_strike_idx = i
        if i == last_heel_strike_idx:
            phase_vec[i] = 0
        else:
            y_vec[i] = y_vec[i - 1] + (thigh_angle_vec[i - 1] - mean_angle) * dt
            val = 100 * np.arctan2((y_vec[i] - mean_y) * x_y_ratio, (x_vec[i] - mean_x)) / (2 * np.pi)  # q_int_q
            period = 100
            val_list = np.array([val - period, val, val + period])
            dist_vec = np.abs(val_list - phase_vec[i - 1])
            phase_vec[i] = val_list[np.argmin(dist_vec)]
    phase_vec = np.clip(phase_vec, 0, 100)
    return phase_vec, x_vec, y_vec


def calc_phase_based_on_q_dq(thigh_angle_vec):
    dq_vec = np.zeros(100)
    dq_vec[1:] = -(thigh_angle_vec[1:100] - thigh_angle_vec[:100 - 1])
    dq_max, dq_min = (np.max(dq_vec), np.min(dq_vec))
    q_max, q_min = (np.max(thigh_angle_vec[:100]), np.min(thigh_angle_vec[:100]))
    mean_x = 0.5 * (q_max + q_min)
    mean_y = 0.5 * (dq_max + dq_min)
    x_y_ratio = (q_max - q_min) / (dq_max - dq_min)
    x_vec = np.zeros(thigh_angle_vec.shape)
    y_vec = np.zeros(thigh_angle_vec.shape)
    phase_vec = np.zeros(thigh_angle_vec.shape)
    last_heel_strike_idx = 0
    for i in range(len(thigh_angle_vec)):
        x_vec[i] = thigh_angle_vec[i]
        if thigh_angle_vec[i] > np.max(thigh_angle_vec) - 1e-2:
            last_heel_strike_idx = i
        if i == last_heel_strike_idx:
            phase_vec[i] = 0
        else:
            y_vec[i] = -(thigh_angle_vec[i] - thigh_angle_vec[i - 1])
            val = 100 * np.arctan2((y_vec[i] - mean_y) * x_y_ratio, (x_vec[i] - mean_x)) / (2 * np.pi)  # q_dq
            period = 100
            val_list = np.array([val - period, val, val + period])
            dist_vec = np.abs(val_list - phase_vec[i - 1])
            phase_vec[i] = val_list[np.argmin(dist_vec)]
    phase_vec = np.clip(phase_vec, 0, 100)
    return phase_vec, x_vec, y_vec


class KalmanFilter(object):
    '''
    Simplified Kalman Filter only deals with the following dynamic model:
    x_k = F_k-1 x_k-1 + w_k
    y_k = H_k x_k + v_k
    '''

    def __init__(self, P, Q, R, F, H, x_0):
        '''
        P: covariance matrix of the state, which indicates the uncertainty of the current estimation
        Q: covariance matrix of the process noises w
        R: covariance matrix of the measures noises v
        F: state transition matrix
        H: observation matrix
        '''
        self.P = P
        self.Q = Q
        self.R = R
        self.F = F
        self.H = H
        self.x = x_0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q  # @ indicates matrix multiplication

    def update(self, y):
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (y - self.H @ self.x)
        I = np.eye(len(self.x))
        self.P = (I - K @ self.H) @ self.P

    def forward(self, y, F):
        self.F = F
        self.predict()
        self.update(y)
        return self.x


class OnlineGaitPredicter(object):
    def __init__(self):
        self.gait_start = False
        self.acc_foot_threshold = 15 # m/s^2
        self.state = 2
        self.gait_paras_dict = np.load('data/paras/gait_paras_dict.npy', allow_pickle=True).item()
        self.kf = self.init_phase_filter()
        self.joint_data_mat = np.zeros((1, 4)) # thigh, knee, ankle, time
        self.phase_vec = np.zeros(2)

    def init_phase_filter(self):
        dt = 1 # 1 represent 10 ms
        R = np.array([[0.5, 0],
                      [0, 5]])  # measurement noise covariance
        P = 0.1 * R  # uncertainty covariance
        Q = np.array([[0, 0],
                      [0, 1e-6]])  # process noise covariance
        F = np.array([[1, dt],
                      [0, 1]])
        H = np.array([[1., 0.],
                      [0., 1.]])
        kf = KalmanFilter(P=P, Q=Q, R=R, F=F, H=H, x_0=np.array([0, 1]))
        return kf


    def forward(self, data_vec, dt = 1):
        data_vec = data_vec.reshape((1, -1))
        roll_vec = data_vec[:, [0, 0 + 9, 0 + 9 * 2]]  # thigh, shank, foot
        acc_z_vec = data_vec[:, [5, 5 + 9, 5 + 9 * 2]]  # thigh, shank, foot
        joint_data_vec = np.zeros(4) # thigh, knee, ankle, time
        joint_data_vec[:3] = calc_joint_angle(roll_vec, is_rad=False) # thigh, knee, ankle
        joint_data_vec[-1] = data_vec[:, -1]
        self.joint_data_mat = np.r_[self.joint_data_mat, joint_data_vec.reshape((1, -1))]
        if self.state > 0 and acc_z_vec[0, -1] > self.acc_foot_threshold:
            self.gait_start = True
            self.state = 0
            self.joint_data_mat = np.zeros((1, 4))  # thigh, knee, ankle, time
            self.joint_data_mat[0] = joint_data_vec
            self.kf = self.init_phase_filter()
            self.phase_vec = np.zeros(2)
        if not self.gait_start:
            return None
        else:
            if len(self.joint_data_mat) > 1:
                thigh_angle_vec = self.joint_data_mat[:, 0]
                phase, self.state, self.gait_paras_dict = estimate_phase_at_one_step(
                    thigh_angle_vec, self.state, self.gait_paras_dict)
                self.phase_vec = fifo_data_vec(self.phase_vec, phase)
                F = np.array([[1, dt],
                              [0, 1]])
                phase = self.kf.forward(np.array([phase, (self.phase_vec[1]-self.phase_vec[0])/dt]), F)[0]
                phase = np.clip(phase, a_min=0, a_max=100)
            else:
                phase = 0
            estimated_joint_angle_vec = self.gait_paras_dict['f_joint_angle'](np.array([phase]))
            return phase, estimated_joint_angle_vec



class OnlinePhasePredicter(object):
    def __init__(self, mode='level_ground'):
        self.gait_start = False
        self.acc_foot_threshold = 15 # m/s^2
        self.state = 2
        self.gait_paras_dict = np.load('data/paras/gait_paras_dict_{}.npy'.format(mode), allow_pickle=True).item()
        self.kf = self.init_phase_filter()
        self.thigh_angle_vec = np.zeros(0)
        self.phase_vec = np.zeros(2)
        self.phase_vec_all = np.zeros(0)

    def init_phase_filter(self):
        dt = 1 # 1 represent 10 ms
        R = np.array([[0.5, 0],
                      [0, 5]])  # measurement noise covariance
        P = 0.1 * R  # uncertainty covariance
        Q = np.array([[0, 0],
                      [0, 1e-6]])  # process noise covariance
        F = np.array([[1, dt],
                      [0, 1]])
        H = np.array([[1., 0.],
                      [0., 1.]])
        kf = KalmanFilter(P=P, Q=Q, R=R, F=F, H=H, x_0=np.array([0, 1]))
        return kf


    def forward(self, thigh_angle, acc_z, dt = 1):
        if self.state == 2 and acc_z > self.acc_foot_threshold:
            self.gait_start = True
            self.state = 0
            self.thigh_angle_vec = np.zeros(0)
            self.kf = self.init_phase_filter()
            self.phase_vec = np.zeros(2)
            self.phase_vec_all = np.zeros(1)
        if not self.gait_start:
            return None
        else:
            self.thigh_angle_vec = np.append(self.thigh_angle_vec, thigh_angle)
            if len(self.thigh_angle_vec) > 1:
                phase, self.state, self.gait_paras_dict = estimate_phase_at_one_step(
                    self.thigh_angle_vec, self.state, self.gait_paras_dict)
                # phase = np.clip(phase, a_min=np.max(self.phase_vec_all), a_max=100)[0]
                self.phase_vec = fifo_data_vec(self.phase_vec, phase)
                F = np.array([[1, dt],
                              [0, 1]])
                phase = self.kf.forward(np.array([phase, (self.phase_vec[1]-self.phase_vec[0])/dt]), F)[0]
                phase = np.clip(phase, a_min=np.max(self.phase_vec_all), a_max=100)[0]
                self.phase_vec_all = np.append(self.phase_vec_all, phase)
            else:
                phase = 0
            return phase




def estimate_phase_at_one_step(thigh_angle_vec, state, paras_dict):
    stance_end_threshold = paras_dict['stance_end_threshold']
    swing_end_threshold = paras_dict['swing_end_threshold']
    popt_list = paras_dict['popt_list']
    idx_list = paras_dict['idx_list']
    if state == 0:
        min_thigh_angle= np.min(thigh_angle_vec)
        if (thigh_angle_vec[-1] <= stance_end_threshold + 1 or
                (thigh_angle_vec[-1] > min_thigh_angle + 2 and
                 min_thigh_angle < stance_end_threshold + 5)):
            state = 1
            paras_dict['stance_end_idx'] = len(thigh_angle_vec)
    else:
        stance_end_idx = paras_dict['stance_end_idx']
        max_thigh_angle = np.max(thigh_angle_vec[stance_end_idx:])
        if (thigh_angle_vec[-1] >= swing_end_threshold - 1 or
                (thigh_angle_vec[-1] < max_thigh_angle - 2 and
                 max_thigh_angle > swing_end_threshold - 5)):
            state = 2
    phase = calc_phase_point(thigh_angle_vec[[-1]], popt_list[state], x_low=idx_list[state], x_high=idx_list[state + 1])
    return phase, state, paras_dict


def estimate_phase_in_real_time(thigh_angle_vec, paras_dict):
    stance_end_threshold = paras_dict['stance_end_threshold']
    swing_end_threshold = paras_dict['swing_end_threshold']
    popt_list = paras_dict['popt_list']
    idx_list = paras_dict['idx_list']
    state = 0
    phase_vec = np.zeros(101)
    state_vec = np.zeros(101)
    angle_varying_threshold = 10
    for i in range(1, 101):
        if state == 0:
            if (thigh_angle_vec[i] < stance_end_threshold or
                    (thigh_angle_vec[i] > np.min(thigh_angle_vec[:i]) + 2 and
                     np.min(thigh_angle_vec[:i]) < stance_end_threshold + angle_varying_threshold)
            ):
                state = 1
                stance_end_idx = i
        else:
            max_thigh_angle = np.max(thigh_angle_vec[stance_end_idx:i])
            if (thigh_angle_vec[i] > swing_end_threshold or
                    (thigh_angle_vec[i] < max_thigh_angle - 2 and
                     np.max(thigh_angle_vec[stance_end_idx:i]) > swing_end_threshold - angle_varying_threshold)):
                state = 2
        state_vec[i] = state
        phase_vec[i] = calc_phase_point(thigh_angle_vec[[i]], popt_list[state],
                                        x_low=idx_list[state], x_high=idx_list[state + 1])
    filtered_phase_vec = filter_gait_phase(phase_vec, dt=1)
    phase_vec[np.isnan(phase_vec)] = 100
    filtered_phase_vec = np.clip(filtered_phase_vec, a_min=0, a_max=100)
    rmse = np.sqrt(mse(np.arange(101), filtered_phase_vec))
    correlation, _ = pearsonr(np.arange(101), filtered_phase_vec)
    return filtered_phase_vec, rmse, correlation, state_vec, phase_vec


def analyze_online_imu_data(data_mat):
    acc_z_mat = data_mat[:, [5, 5 + 9, 5 + 9 * 2]]
    time_vec = data_mat[:, -1]

    # plt.plot(time_vec[1:] - time_vec[:-1])
    # plt.show()

    acc_threshold = 15
    gait_event_indices = filter_gait_event_indices(np.where(acc_z_mat[:, 2] > acc_threshold)[0])

    predicted_phase_vec = np.zeros(time_vec.shape)
    predictor = OnlineGaitPredicter()
    for i in range(gait_event_indices[0], gait_event_indices[-1] - 1):
        time_start = time.time()
        predicted_results = predictor.forward(data_mat[i], dt = 100 * (time_vec[i] - time_vec[i-1]))
        if predicted_results is None:
            continue
        else:
            predicted_phase_vec[i], _ = predicted_results
        print('Running time: {:.3f} s'.format(time.time() - time_start))

    actual_phase_vec = np.zeros(time_vec.shape)
    for i in range(gait_event_indices[0], gait_event_indices[-1]-1):
        dist = gait_event_indices - i
        gait_start_idx = gait_event_indices[dist<=0][-1]
        gait_end_idx = gait_event_indices[dist > 0][0]
        actual_phase_vec[i] = 100 * (time_vec[i] - time_vec[gait_start_idx])/(time_vec[gait_end_idx] - time_vec[gait_start_idx])

    plt.plot(predicted_phase_vec)
    plt.plot(actual_phase_vec, '--')
    rmse_val = rmse(predicted_phase_vec[gait_event_indices[0]:gait_event_indices[-1]],
         actual_phase_vec[gait_event_indices[0]:gait_event_indices[-1]], axis=0)
    r_val, _ = pearsonr(predicted_phase_vec[gait_event_indices[0]:gait_event_indices[-1]],
         actual_phase_vec[gait_event_indices[0]:gait_event_indices[-1]])
    title_str = 'RMSE: {:.3f}, r: {:.3f}'.format(rmse_val, r_val)
    print(title_str)
    plt.title(title_str)
    image_name = 'analyze_online_imu_data'
    plt.savefig('{}/{}.pdf'.format('results/images', image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format('results/images', image_name), bbox_inches='tight', dpi=300)
    plt.show()


def analyze_offline_outdoor_data(signal_folder = 'data/IMUOut1'):
    # signal_with_label_vec = IO.read_signals_with_label(signal_folder)
    signal_with_label_vec = np.load('{}/signal_with_label_vec.npy'.format(signal_folder))
    print(signal_with_label_vec.shape)
    '''
        1. Segment gait
    '''
    acc_z_mat = signal_with_label_vec[:, [2 + 9 + 5, 2 + 9 * 2 + 5, 2 + 9 * 3 + 5]]
    roll_mat = signal_with_label_vec[:, [2 + 9, 2 + 9 * 2, 2 + 9 * 3]]
    roll_mat[:, -1] *= -1 # the axis direction of the foot is opposite to the that of knee and ankle
    if 'Eve' in signal_folder:
        threshold_dict = {'acc_threshold': 6, 'idx_diff_threshold': 50}
    else:
        threshold_dict = {'acc_threshold': 17, 'idx_diff_threshold': 30}
    gait_event_indices = filter_gait_event_indices(np.where(acc_z_mat[:, 2] > threshold_dict['acc_threshold'])[0],
                                                   idx_diff_threshold=threshold_dict['idx_diff_threshold'])
    # Plot.plot_gait_segmentation(roll_mat, gait_event_indices, acc_z_mat, threshold_dict['acc_threshold'])
    '''
        2. Plot normalized thigh angle
    '''
    steps = len(gait_event_indices) - 1
    joint_angle_mat = np.zeros((steps, 101, 3))  # (steps, gait cycle, joint number)
    gait_label_vec = np.zeros(steps)
    captured_joint_angle_mat = calc_joint_angle(roll_mat, is_rad=False)
    time_vec = signal_with_label_vec[:, 0]
    label_vec = signal_with_label_vec[:, 1]
    for i in range(steps):
        step_indices = np.arange(gait_event_indices[i], gait_event_indices[i + 1])
        joint_angle_vec_i = captured_joint_angle_mat[step_indices]
        gait_label_vec[i] = np.median(label_vec[step_indices])
        time_vec_i = time_vec[step_indices]
        joint_angle_mat[i] = interpolate_joint_angle(joint_angle_vec_i, time_vec_i)

    joint_angle_mat, gait_label_vec = remove_outliers_of_joint_angle_mat(joint_angle_mat, gait_label_vec, std_ratio=2)

    # for i in range(len(joint_angle_mat)):
    #     joint_angle_mat[i, :, 0] = filter_gait_phase(joint_angle_mat[i, :, 0], dt=1)

    # Plot.plot_segmented_joint_angles(joint_angle_mat, gait_label_vec)
    '''
    3. Calculate the global parameters: gait cycle, phase variable parameters,
    '''
    label_name_list = ['level_ground', 'stair_ascent', 'stair_descent', 'ramp_ascent', 'ramp_descent']
    joint_angle_mat_list = []
    estimated_joint_angle_mean_list = []
    for k in range(5):
        joint_angle_mat_k = joint_angle_mat[gait_label_vec == k]
        joint_angle_mat_k[..., 1] *= -1 # change the sign of the knee angle
        joint_angle_mat_list.append(joint_angle_mat_k)
        joint_angle_mean = np.mean(joint_angle_mat_k, axis=0)
        joint_angle_mean = filter_joint_angle_mean(joint_angle_mean)
        thigh_angle_mean = joint_angle_mean[:, 0]
        # thigh_angle_mean = np.median(joint_angle_mat_k[..., 0], axis=0)
        gait_paras_dict = calc_phase_parameters(thigh_angle_mean)
        f_joint_angle = interpolate.interp1d(np.arange(start = 0, stop = 101, step = 5), joint_angle_mean[::5],
                                             kind='cubic', axis=0)
        estimated_joint_angle_mean_list.append(f_joint_angle(np.arange(101)))

        gait_paras_dict['joint_angle_mean'] = joint_angle_mean
        gait_paras_dict['f_joint_angle'] = f_joint_angle
        np.save('data/paras/gait_paras_dict_{}.npy'.format(label_name_list[k]), gait_paras_dict)

        '''
            4. Estimate all phase variables.
        '''
        filtered_phase_mat = np.zeros(joint_angle_mat_k.shape[:2])
        phase_mat = np.zeros(joint_angle_mat_k.shape[:2])
        state_mat = np.zeros(joint_angle_mat_k.shape[:2])
        rmse_vec = np.zeros((len(filtered_phase_mat)))
        r_vec = np.zeros((len(filtered_phase_mat)))

        for i in range(len(filtered_phase_mat)):
            filtered_phase_mat[i], rmse_vec[i], r_vec[i], state_mat[i], phase_mat[i] = estimate_phase_in_real_time(
                thigh_angle_vec=joint_angle_mat_k[i, :, 0], paras_dict=gait_paras_dict)

        estimated_joint_angle_mat = f_joint_angle(filtered_phase_mat.reshape(-1))

        desired_phase_vec = np.repeat(np.arange(101).reshape((1, -1)), repeats=len(filtered_phase_mat), axis=0).reshape(
            -1)
        estimated_joint_angle_mat[:, 0] = joint_angle_mat_k.reshape((-1, 3))[:, 0]

        Plot.plot_estimated_phase_and_angle(filtered_phase_mat, desired_phase_vec, estimated_joint_angle_mat,
                                            joint_angle_mat_k, label_name_list[k])

    Plot.plot_estimated_joint_angles(joint_angle_mat_list, estimated_joint_angle_mean_list)


def filter_joint_angle_mean(joint_angle_mean):
    joint_angle_mean_long = np.r_[joint_angle_mean, joint_angle_mean[1:],joint_angle_mean[1:]]
    joint_angle_mean_long_filter = np.copy(joint_angle_mean_long)
    for i in range(3):
        joint_angle_mean_long_filter[:, i] = sp.signal.savgol_filter(joint_angle_mean_long[:, i],15,2)
    plt.plot(joint_angle_mean_long, '--')
    plt.plot(joint_angle_mean_long_filter)
    plt.show()
    return joint_angle_mean_long_filter[100:201]


def analyze_offline_imu_data(data_mat):
    roll_mat = data_mat[:, [0, 0 + 9, 0 + 9 * 2]]
    acc_z_mat = data_mat[:, [5, 5 + 9, 5 + 9 * 2]]
    time_vec = data_mat[:, -1]
    '''
    1. Segment gait
    '''
    acc_threshold = 15
    gait_event_indices = filter_gait_event_indices(np.where(acc_z_mat[:, 2] > acc_threshold)[0])
    print(gait_event_indices[1:] - gait_event_indices[:-1])
    '''
    2. Plot normalized thigh angle
    '''
    steps = len(gait_event_indices) - 1
    joint_angle_mat = np.zeros((steps, 101, 3))  # (steps, gait cycle, joint number)
    captured_joint_angle_mat = calc_joint_angle(roll_mat, is_rad=False)
    for i in range(steps):
        step_indices = np.arange(gait_event_indices[i], gait_event_indices[i + 1])
        joint_angle_vec_i = captured_joint_angle_mat[step_indices]
        time_vec_i = time_vec[step_indices]
        joint_angle_mat[i] = interpolate_joint_angle(joint_angle_vec_i, time_vec_i)

    '''
    3. Calculate the global parameters: gait cycle, phase variable parameters,
    '''
    thigh_angle_mean = np.mean(joint_angle_mat[..., 0], axis=0)
    gait_paras_dict = calc_phase_parameters(thigh_angle_mean)
    joint_angle_mean = np.mean(joint_angle_mat, axis=0)
    f_joint_angle = interpolate.interp1d(np.arange(101), joint_angle_mean,
                                         kind='cubic', axis=0)
    gait_paras_dict['joint_angle_mean'] = joint_angle_mean
    gait_paras_dict['f_joint_angle'] = f_joint_angle
    np.save('data/paras/gait_paras_dict.npy', gait_paras_dict)
    '''
    4. Estimate all phase variables.
    '''
    filtered_phase_mat = np.zeros(joint_angle_mat.shape[:2])
    phase_mat = np.zeros(joint_angle_mat.shape[:2])
    state_mat = np.zeros(joint_angle_mat.shape[:2])
    rmse_vec = np.zeros((len(filtered_phase_mat)))
    r_vec = np.zeros((len(filtered_phase_mat)))

    for i in range(len(filtered_phase_mat)):
        filtered_phase_mat[i], rmse_vec[i], r_vec[i], state_mat[i], phase_mat[i] = estimate_phase_in_real_time(
            thigh_angle_vec=joint_angle_mat[i, :, 0], paras_dict=gait_paras_dict)
    print('Phase RMSE: {:.3f}, r: {:.3f}'.format(np.mean(rmse_vec), np.mean(r_vec)))

    estimated_joint_angle_mat = f_joint_angle(filtered_phase_mat.reshape(-1))
    rmse_knee = rmse(estimated_joint_angle_mat[:, 1], joint_angle_mat.reshape((-1, 3))[:, 1], axis=0)
    r_knee,_ = pearsonr(estimated_joint_angle_mat[:, 1], joint_angle_mat.reshape((-1, 3))[:, 1])
    rmse_ankle = rmse(estimated_joint_angle_mat[:, 2], joint_angle_mat.reshape((-1, 3))[:, 2], axis=0)
    r_ankle, _ = pearsonr(estimated_joint_angle_mat[:, 2], joint_angle_mat.reshape((-1, 3))[:, 2])
    print('Knee RMSE: {:.3f}, r: {:.3f}'.format(rmse_knee, r_knee))
    print('Ankle RMSE: {:.3f}, r: {:.3f}'.format(rmse_ankle, r_ankle))

    desired_phase_vec = np.repeat(np.arange(101).reshape((1, -1)), repeats=len(filtered_phase_mat), axis=0).reshape(
        -1)

    phase_vec_list = [filtered_phase_mat.reshape(-1), desired_phase_vec]
    estimated_joint_angle_mat[:, 0] = joint_angle_mat.reshape((-1, 3))[:, 0]
    joint_angle_vec_list = [estimated_joint_angle_mat, joint_angle_mat.reshape((-1, 3))]
    # ----------Figure-----------
    img_dir = 'results/images'
    fig = plt.figure(figsize=(8, 2.5))
    x = np.arange(-10, 110)
    cm_fun = cm.get_cmap('tab20', 20)
    idx_list = gait_paras_dict['idx_list']
    popt_list = gait_paras_dict['popt_list']
    for i in range(3):
        plt.plot(np.arange(idx_list[i], idx_list[i+1]), thigh_angle_mean[idx_list[i]:idx_list[i+1]], 'o',
                 markersize=5, color = cm_fun(2*i+1))
        plt.plot(x, fun_x_to_y(x, *popt_list[i]), color = cm_fun(2*i))
    plt.xlabel(r'Gait phase $s$ (%)')
    plt.ylabel(r'Thigh angle $\phi$ ($\degree$)')
    plt.xlim([-5, 105])
    # plt.legend(['Stance: measured', 'Stance: fitted',
    #             'Swing: measured', 'Swing: fitted',
    #             'Late swing: measured', 'Late swing: fitted',])
    plt.legend(['Measured', 'Fitted'], ncol=2)
    fig.tight_layout()
    image_name = 'fitted_thigh_angle'
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()

    # fig = plt.figure(figsize=(6, 6))
    # plt.subplot(2, 1, 1)
    # x = np.arange(-10, 110)
    # cm_fun = cm.get_cmap('tab20', 20)
    # idx_list = gait_paras_dict['idx_list']
    # popt_list = gait_paras_dict['popt_list']
    # start_idx = 0
    # plt.hlines(y=[gait_paras_dict['stance_end_threshold'], gait_paras_dict['swing_end_threshold']], xmin=0, xmax=100)
    # for i in range(3):
    #     thigh_angle_vec = joint_angle_mat[0, state_mat[0] == i, 0]
    #     plt.plot(np.arange(start_idx, start_idx+len(thigh_angle_vec)), thigh_angle_vec, 'o',
    #              markersize=5, color=cm_fun(2 * i + 1))
    #     plt.plot(x, fun_x_to_y(x, *popt_list[i]), '--', color=cm_fun(2 * i))
    #     start_idx = len(thigh_angle_vec)
    # plt.xticks([])
    # plt.ylabel(r'Thigh angle $\phi$ ($\degree$)')
    # plt.subplot(2, 1, 2)
    # start_idx = 0
    # for i in range(3):
    #     phase_vec = phase_mat[0, state_mat[0]==i]
    #     plt.plot(np.arange(start_idx, start_idx+len(phase_vec)), phase_vec, color=cm_fun(2 * i))
    #     start_idx = len(phase_vec)
    # plt.xlabel(r'Gait phase $s$ (%)')
    # plt.ylabel(r'Predicted gait phase $\hat{s}$ (%)')
    # # plt.legend(['Stance: measured', 'Stance: fitted',
    # #             'Swing: measured', 'Swing: fitted',
    # #             'Late swing: measured', 'Late swing: fitted', ])
    # fig.tight_layout()
    # image_name = 'thigh_angle_to_phase'
    # plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    # plt.show()

    # fig = plt.figure(figsize=(8, 5))
    # cm_fun = cm.get_cmap('tab10', 10)
    # start_idx = 0
    # plt.subplot(2, 1, 1)
    # plt.plot(np.arange(0, 101), np.arange(0, 101), '--', color=cm_fun(0))
    # plt.plot(np.arange(0, 101), phase_mat[0], color=cm_fun(1))
    # rmse_val = rmse(np.arange(101), phase_mat[0])
    # r_val,_ = pearsonr(np.arange(101), phase_mat[0])
    # plt.xticks([])
    # plt.ylabel(r'Predicted gait phase $\hat{s}$ (%)')
    # plt.title('RMSE: {:.3f}, r: {:.3f}'.format(rmse_val, r_val))
    # plt.legend(['Actual', 'Predicted'], ncol=2)
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(0, 101), np.arange(0, 101), '--', color=cm_fun(0))
    # plt.plot(np.arange(0, 101), filtered_phase_mat[0], color=cm_fun(2))
    # plt.xlabel(r'Gait phase $s$ (%)')
    # plt.ylabel(r'Filtered gait phase $\hat{s}$ (%)')
    # plt.legend(['Actual', 'Filtered'], ncol=2)
    # rmse_val = rmse(np.arange(101), filtered_phase_mat[0])
    # r_val, _ = pearsonr(np.arange(101), filtered_phase_mat[0])
    # plt.title('RMSE: {:.3f}, r: {:.3f}'.format(rmse_val, r_val))
    # fig.tight_layout()
    # image_name = 'filtered_phase_vec'
    # plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    # plt.show()

    fig = plt.figure(figsize=(8, 2.5))
    cm_fun = cm.get_cmap('tab10', 10)
    start_idx = 0
    # plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, 101), np.arange(0, 101), '-.', color=cm_fun(0), zorder=1)
    plt.plot(np.arange(0, 101), phase_mat[0], '--', color=cm_fun(1), zorder=3)
    plt.plot(np.arange(0, 101), filtered_phase_mat[0], color=cm_fun(2), zorder=2)
    rmse_val = rmse(np.arange(101), phase_mat[0])
    r_val, _ = pearsonr(np.arange(101), phase_mat[0])
    plt.xlabel(r'Gait phase $s$ (%)')
    plt.ylabel(r'Predicted gait phase $\hat{s}$ (%)')
    # plt.title('RMSE: {:.3f}, r: {:.3f}'.format(rmse_val, r_val))
    plt.legend(['Actual', 'Predicted', 'Filtered'], ncol=3)
    # plt.subplot(2, 1, 2)
    # plt.plot(np.arange(0, 101), np.arange(0, 101), '--', color=cm_fun(0))
    # plt.plot(np.arange(0, 101), filtered_phase_mat[0], color=cm_fun(2))
    # plt.xlabel(r'Gait phase $s$ (%)')
    # plt.ylabel(r'Filtered gait phase $\hat{s}$ (%)')
    # plt.legend(['Actual', 'Filtered'], ncol=2)
    # rmse_val = rmse(np.arange(101), filtered_phase_mat[0])
    # r_val, _ = pearsonr(np.arange(101), filtered_phase_mat[0])
    # plt.title('RMSE: {:.3f}, r: {:.3f}'.format(rmse_val, r_val))
    fig.tight_layout()
    image_name = 'filtered_phase_vec'
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()

    # fig = plt.figure(figsize=(16, 7))
    # plt.subplot(2, 1, 1)
    # plt.plot(roll_mat)
    # plt.ylabel(r'Angle ($\degree$)')
    # plt.vlines(gait_event_indices, ymin=-50, ymax=50)
    # plt.subplot(2, 1, 2)
    # plt.plot(acc_z_mat)
    # plt.hlines(acc_threshold, gait_event_indices[0], gait_event_indices[-1])
    # plt.ylabel(r'Z-axis Acceleration (m/s^2)')
    # plt.xlabel('Time steps')
    # fig.tight_layout()
    # fig.legend(['Thigh', 'Shank', 'Foot', 'Heel strike'], loc='lower center',
    #            ncol=4, bbox_to_anchor=(0.50, 0.97), frameon=False)

    # image_name = 'imu_data_segmentation'
    # plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    # plt.show()
    # fig = plt.figure(figsize=(16, 7))
    # ylabel_list = ['Thigh', 'Knee', 'Ankle']
    # for j in range(3):
    #     plt.subplot(3, 2, 2 * j + 1)
    #     plt.plot(joint_angle_mat[..., j].T)
    #     plt.ylabel(r'{} angle ($\degree$)'.format(ylabel_list[j]))
    #     if 2 == j:
    #         plt.xlabel('Gait phase (%)')
    #     plt.subplot(3, 2, 2 * j + 2)
    #     plt.plot(np.mean(joint_angle_mat[..., j], axis=0))
    #     if 2 == j:
    #         plt.xlabel('Gait phase (%)')
    #
    # fig.tight_layout()
    # image_name = 'segmented_imu_data'
    # plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    # plt.show()


    # fig = plt.figure(figsize=(16, 7))
    # plt.subplot(4, 1, 1)
    # plt.plot(phase_mat.reshape(-1))
    # plt.plot(desired_phase_vec, '--')
    # plt.ylabel('Gait phase (%)')
    # plt.xticks([])
    # ylabel_list = ['Thigh', 'Knee', 'Ankle']
    # for j in range(3):
    #     plt.subplot(4, 1, j+2)
    #     plt.plot(estimated_joint_angle_mat[:, j])
    #     plt.plot(joint_angle_mat[..., j].reshape(-1), '--')
    #     if j < 2:
    #         plt.xticks([])
    #     plt.ylabel(r'{} angle ($\degree$)'.format(ylabel_list[j]))
    # plt.xlabel('Time steps')
    # fig.legend(['Predicted', 'Actual'], loc='lower center',
    #            ncol=2, bbox_to_anchor=(0.50, 0.97), frameon=False)
    # fig.tight_layout()
    # image_name = 'fitted_imu_data'
    # plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    # plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    # plt.show()
    # ------------------------------------
    return phase_vec_list, joint_angle_vec_list







if __name__ == '__main__':
    a = 1
