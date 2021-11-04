import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})
import os
import glob
import open3d as o3d
from Utils import Algo, IO
from matplotlib import cm
from scipy.stats import pearsonr
from scipy import interpolate


def plot_estimated_phase_and_angle(phase_mat, desired_phase_vec, estimated_joint_angle_mat, joint_angle_mat,
                                   label_name):
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(4, 1, 1)
    plt.plot(phase_mat.reshape(-1))
    plt.plot(desired_phase_vec, '--')
    rmse_vec = np.zeros(phase_mat.shape[0])
    r_vec = np.zeros(phase_mat.shape[0])
    for i in range(phase_mat.shape[0]):
        rmse_vec[i] = Algo.rmse(phase_mat[i], desired_phase_vec.reshape((phase_mat.shape))[i], axis=0)
        r_vec[i], _ = pearsonr(phase_mat[i], desired_phase_vec.reshape((phase_mat.shape))[i])
    plt.title('RMSE: {:.3f}, R: {:.3f}'.format(np.mean(rmse_vec), np.mean(r_vec)))

    plt.ylabel('Gait phase (%)')
    plt.xticks([])
    ylabel_list = ['Thigh', 'Knee', 'Ankle']
    for j in range(3):
        plt.subplot(4, 1, j + 2)
        plt.plot(estimated_joint_angle_mat[:, j])
        plt.plot(joint_angle_mat[..., j].reshape(-1), '--')
        for i in range(phase_mat.shape[0]):
            rmse_vec[i] = Algo.rmse(estimated_joint_angle_mat[:, j].reshape(phase_mat.shape)[i], joint_angle_mat[i, :, j], axis=0)
            r_vec[i], _ = pearsonr(estimated_joint_angle_mat[:, j].reshape(phase_mat.shape)[i], joint_angle_mat[i, :, j])
        plt.title('RMSE: {:.3f}, R: {:.3f}'.format(np.mean(rmse_vec), np.mean(r_vec)))
        if j < 2:
            plt.xticks([])
        plt.ylabel(r'{} angle ($\degree$)'.format(ylabel_list[j]))
    plt.xlabel('Time steps\n{}'.format(label_name))
    fig.legend(['Predicted', 'Actual'], loc='lower center',
               ncol=2, bbox_to_anchor=(0.50, 0.97), frameon=False)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'fitted_outdoor_{}_data'.format(label_name)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_estimated_joint_angles(joint_angle_mat_list, estimated_joint_angle_mean_list):
    label_name_list = ['Level ground', 'Stair ascent', 'Stair descent', 'Ramp ascent', 'Ramp descent']
    fig = plt.figure(figsize=(16, 7))
    cm_fun = cm.get_cmap('tab10', 10)
    for i in range(5):
        joint_angle_mat_i = joint_angle_mat_list[i]
        ylabel_list = ['Thigh', 'Knee', 'Ankle']
        for j in range(3):
            plt.subplot(3, 5, 5 * j + i + 1)
            joint_angle_mean_i = np.mean(joint_angle_mat_i[..., j], axis=0)
            joint_angle_std_i = np.std(joint_angle_mat_i[..., j], axis=0)
            plt.plot(estimated_joint_angle_mean_list[i][..., j], color = cm_fun(1))
            plt.plot(joint_angle_mean_i, '--', color = cm_fun(0))
            plt.fill_between(np.arange(joint_angle_mean_i.shape[0]),
                             joint_angle_mean_i - joint_angle_std_i,
                             joint_angle_mean_i + joint_angle_std_i,
                             alpha=0.3, color = cm_fun(0))
            if i == 0:
                plt.ylabel(r'{} angle ($\degree$)'.format(ylabel_list[j]))
            if 2 == j:
                plt.xlabel('Gait phase (%)\n{}'.format(label_name_list[i]))
            # plt.subplot(3, 2, 2 * j + 2)
            # plt.plot(np.mean(joint_angle_mat_i[..., j], axis=0))
            # plt.ylim([np.min(joint_angle_mat_i[..., j])-5, np.max(joint_angle_mat_i[..., j])+5])
            if 2 == j:
                plt.xlabel('Gait phase (%)\n{}'.format(label_name_list[i]))
    fig.legend(['Predicted', 'Actual'], loc='lower center',
               ncol=2, bbox_to_anchor=(0.50, 0.97), frameon=False)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'outdoor_estimated_angle_mean_and_std'
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_segmented_joint_angles(joint_angle_mat, gait_label_vec):
    label_name_list = ['Level ground', 'Stair ascent', 'Stair descent', 'Ramp ascent', 'Ramp descent']
    for k in range(2):
        fig = plt.figure(figsize=(16, 7))
        for i in range(5):
            joint_angle_mat_i = joint_angle_mat[gait_label_vec == i]
            ylabel_list = ['Thigh', 'Knee', 'Ankle']
            for j in range(3):
                plt.subplot(3, 5, 5 * j + i + 1)
                if k == 0:
                    plt.plot(joint_angle_mat_i[..., j].T)
                else:
                    joint_angle_mean_i = np.mean(joint_angle_mat_i[..., j], axis=0)
                    joint_angle_std_i = np.std(joint_angle_mat_i[..., j], axis=0)
                    plt.plot(joint_angle_mean_i)
                    plt.fill_between(np.arange(joint_angle_mean_i.shape[0]),
                                     joint_angle_mean_i - joint_angle_std_i,
                                     joint_angle_mean_i + joint_angle_std_i,
                                     alpha=0.3)
                if i == 0:
                    plt.ylabel(r'{} angle ($\degree$)'.format(ylabel_list[j]))
                if 2 == j:
                    plt.xlabel('Gait phase (%)\n{}'.format(label_name_list[i]))
                # plt.subplot(3, 2, 2 * j + 2)
                # plt.plot(np.mean(joint_angle_mat_i[..., j], axis=0))
                # plt.ylim([np.min(joint_angle_mat_i[..., j])-5, np.max(joint_angle_mat_i[..., j])+5])
                if 2 == j:
                    plt.xlabel('Gait phase (%)\n{}'.format(label_name_list[i]))
        fig.tight_layout()
        img_dir = 'results/images'
        if k == 0:
            image_name = 'outdoor_segmented_angle'
        else:
            image_name = 'outdoor_segmented_angle_mean_and_std'
        plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
        plt.show()


def plot_gait_segmentation(roll_mat, gait_event_indices, acc_z_mat, acc_threshold):
    fig = plt.figure(figsize=(16, 7))
    plt.subplot(2, 1, 1)
    plt.plot(roll_mat[:, 0])
    plt.ylabel(r'Thigh angle ($\degree$)')
    plt.vlines(gait_event_indices, ymin=-100, ymax=100)
    plt.legend(['Angle curve', 'Gait event'], loc='lower center', ncol=2, bbox_to_anchor=(0.50, 0.95),
               frameon=False)
    plt.subplot(2, 1, 2)
    plt.plot(acc_z_mat[:, 2], zorder=1)
    plt.hlines(acc_threshold, gait_event_indices[0] - 100, gait_event_indices[-1] + 100, zorder=2)
    plt.ylabel(r'Foot z-axis acceleration (m/s^2)')
    plt.xlabel('Time steps')
    plt.legend(['Acceleration', 'Heel strike threshold'], loc='lower center', ncol=2, bbox_to_anchor=(0.50, 0.95),
               frameon=False)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'outdoor_gait_segmentation'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def init_o3d_vis(width=640, height=480):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=True)
    return vis


def view_o3d_vis(vis, pcd_list, viewer_setting_file=None, img_name=None):
    # best view status
    vis.clear_geometries()
    for init_pcd in pcd_list:
        vis.add_geometry(init_pcd)
    if viewer_setting_file is not None:
        vis = IO.load_view_point(vis, viewer_setting_file)
    vis.poll_events()
    vis.update_renderer()
    if img_name is not None:
        vis.capture_screen_image(img_name, do_render=False)
    return vis


def plot_joint_angle_and_cubit_fitting_one_figure():
    '''
            joint_angle_mean_dict: {'stance', 'swing'}
            joint_angle_mean_dict['stance']: [mode number]
            joint_angle_mean_dict['stance'][0]: (101, joint number, types)
        '''
    plot_data = np.load('data/processed/plot_data.npy', allow_pickle=True).item()
    joint_angle_mean_list = plot_data['joint_angle_mean_dict']['all']
    joint_angle_mat = joint_angle_mean_list[0][:, 1:, 1]

    time_vec = np.arange(0, 101)
    step = 5
    joint_angle_mat_new = Algo.interpolate_joint_angle(joint_angle_mat[::step], time_vec[::step], number=101)
    joint_name_list = ['Knee', 'Ankle']
    fig = plt.figure(figsize=(8, 2.5))
    cm_fun = cm.get_cmap('tab10', 10)
    for i in range(1):
        plt.plot(joint_angle_mat[:, i], color=cm_fun(3 * i))
        plt.plot(joint_angle_mat_new[:, i], '--', color=cm_fun(3 * i+1))
        plt.plot(time_vec[::step], joint_angle_mat[::step][:, i], 'o', color=cm_fun(3 * i+2))
        plt.ylabel(r'{} angle $(\degree)$'.format(joint_name_list[i]))
        plt.xlabel('Gait phase (%)')
    fig.legend(['Actual', 'Fitted',  'Knot points'], loc='lower center', ncol=3, bbox_to_anchor=(0.34, 0.3),
               frameon=True)
    # fig.legend(['Actual', 'Fitted', 'Knot points'], loc='best', ncol=3, frameon=True)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'cubic_fitting'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_joint_angle_and_cubit_fitting():
    '''
            joint_angle_mean_dict: {'stance', 'swing'}
            joint_angle_mean_dict['stance']: [mode number]
            joint_angle_mean_dict['stance'][0]: (101, joint number, types)
        '''
    plot_data = np.load('data/processed/plot_data.npy', allow_pickle=True).item()
    joint_angle_mean_list = plot_data['joint_angle_mean_dict']['all']
    joint_angle_mat = joint_angle_mean_list[0][:, 1:, 1]

    time_vec = np.arange(0, 101)
    step = 5
    joint_angle_mat_new = Algo.interpolate_joint_angle(joint_angle_mat[::step], time_vec[::step], number=101)
    joint_name_list = ['Knee', 'Ankle']
    fig = plt.figure(figsize=(5, 9))
    cm_fun = cm.get_cmap('tab10', 10)
    for i in range(2):
        plt.subplot(2, 1, i + 1)
        plt.plot(joint_angle_mat_new[:, i], color=cm_fun(0))
        plt.plot(joint_angle_mat[:, i], '--', color=cm_fun(1))
        plt.plot(time_vec[::step], joint_angle_mat[::step][:, i], 'o', color=cm_fun(2))
        plt.ylabel(r'{} angle $(\degree)$'.format(joint_name_list[i]))
        plt.xlabel('Gait phase (%)')
    fig.legend(['Fitted', 'Actual', 'Knot points'], loc='lower center', ncol=3, bbox_to_anchor=(0.50, 0.96),
               frameon=False)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'cubic_fitting'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_kalman_filter():
    time_list = [1000, 2000, 3000]
    fig = plt.figure(figsize=(16, 6))
    cm_fun = cm.get_cmap('tab10', 10)
    for time in time_list:
        dt = 1
        t_vec = np.arange(0, time, dt)
        x = np.arange(0, 100, step=100 / (time / dt))
        y = x + np.random.normal(0, 5, x.shape)
        R = np.array([[10, 0],
                      [0, 1]])
        F = np.array([[1, dt],
                      [0, 1]])
        Q = np.array([[0, 0],
                      [0, 1e-6]])
        # initialization
        P = np.array([[10, 0],
                      [0, 1]])
        H = np.array([[1., 0.],
                      [0., 1.]])
        plt.plot(t_vec, y, color=cm_fun(0), alpha=0.5)
        y[:10] = x[:10] * 0.9
        x_predict = Algo.filter_measurements(np.copy(y), P, Q, R, F, H)[:, 0]
        plt.plot(t_vec, x_predict, color=cm_fun(1), linewidth=2)
        plt.plot(t_vec, x, '--', color=cm_fun(2), linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Gait phase (%)')
        plt.legend(['Measurement', 'Prediction', 'Actual'])
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'test_Kalman'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_limit_cycle(joint_angle_and_velocity_mat, gait_mode_vec):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(13, 6))
    for i in range(len(mode_list)):
        mode = mode_list[i]
        plt.subplot(2, 3, i + 1)
        plt.plot(joint_angle_and_velocity_mat[gait_mode_vec == mode, 0],
                 joint_angle_and_velocity_mat[gait_mode_vec == mode, 2])
        plt.xlabel('Thigh angle (deg)\n{}'.format(mode_list[i]))
        plt.ylabel('Thigh angular velocity (deg/s)')
        plt.xlim([-50, 60])
        plt.ylim([-300, 350])
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'original_limit_cycle'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_normalized_limit_cycle(all_joint_data, image_name='q_dq', all_condition=None):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 3))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    for i in range(len(mode_list)):
        plt.subplot(1, len(mode_list), i + 1)
        if all_condition is None:
            plt.plot(all_joint_data[i][:, :, 2].reshape((-1, 1)), all_joint_data[i][:, :, 3].reshape((-1, 1)))
        else:
            for j in range(len(np.unique(all_condition[i]))):
                plt.plot(all_joint_data[i][all_condition[i] == j, :, 2].reshape((-1, 1)),
                         all_joint_data[i][all_condition[i] == j, :, 3].reshape((-1, 1)))
            plt.legend(legend_list[i], frameon=False, handlelength=0.5)
        plt.xlabel('Thigh angle (deg)\n{}'.format(mode_list[i]))
        if i == 0:
            plt.ylabel(r'Thigh angular velocity $\dot{\phi}$ (deg/s)')
        plt.xlim([-50, 100])
        plt.ylim([-300, 400])
    fig.tight_layout()
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_phase_cycle(all_joint_data, image_name='q_int_q', all_condition=None):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 5))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    # for r in range(1):
    for r in [0]:
        for i in range(len(mode_list)):
            plt.subplot(1, len(mode_list), i + 1)
            if all_condition is None:
                for c in range(all_joint_data[i].shape[0]):
                    plt.plot(all_joint_data[i][c, :, 2].reshape((-1, 1)),
                             all_joint_data[i][c, :, 4 - r].reshape((-1, 1)))
            else:
                for j in range(len(np.unique(all_condition[i]))):
                    plt.plot(all_joint_data[i][all_condition[i] == j, :, 2].reshape((-1, 1)),
                             all_joint_data[i][all_condition[i] == j, :, 4 - r].reshape((-1, 1)))
                # plt.legend(legend_list[i], frameon=False, handlelength = 0.5)
            plt.xlabel('Thigh angle (deg)\n{}'.format(mode_list[i]))
            if i == 0:
                if r == 0:
                    plt.ylabel(r'Thigh angle integration $\Phi$ (deg$\cdot$s)')
                else:
                    plt.ylabel(r'Thigh angular velocity $\dot{\phi}$ (deg/s)')
            # plt.xlim([-50, 100])
            # plt.ylim([-900, 700])
    fig.tight_layout()
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_gait_phase(all_gait_phase, correlation_coefficient_mat, rmse_mat, image_name='all_gait_phase', method_idx=0):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(13, 10))
    for r in range(len(all_gait_phase)):
        for c in range(len(mode_list)):
            plt.subplot(2, len(mode_list), r * (len(mode_list)) + c + 1)
            # plt.plot(all_gait_phase[c][:, :100, 0], all_gait_phase[c][:, :100, r + 1])
            for k in range(all_gait_phase[r][c].shape[0]):
                plt.plot(all_gait_phase[r][c][k, :100, 0], all_gait_phase[r][c][k, :100, method_idx + 1])
                plt.title('r = {:.3f}, rmse = {:.3f}'.format(
                    correlation_coefficient_mat[r][c, method_idx], rmse_mat[r][c, method_idx]))
            if 1 == r:
                plt.xlabel('Actual gait phase (%)\n{}'.format(mode_list[c]))
            if 0 == c:
                if 0 == r:
                    plt.ylabel('Filtered gait phase (%)')
                else:
                    plt.ylabel('Predicted gait phase (%)')
    fig.tight_layout()
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_mean_points(all_joint_data, all_condition, image_name='mean_angle'):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 3))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    for i in range(len(mode_list)):
        plt.subplot(1, len(mode_list), i + 1)
        if all_condition is None:
            mean_angle = np.mean(all_joint_data[i][:, :, 2], axis=-1)
            plt.plot(all_joint_data[i][:, 0, 2], mean_angle, '.')
        else:
            for j in range(len(np.unique(all_condition[i]))):
                mean_angle = np.mean(all_joint_data[i][all_condition[i] == j, :, 2], axis=-1)
                plt.plot(all_joint_data[i][all_condition[i] == j, 0, 2], mean_angle, '.')
            plt.legend(legend_list[i], frameon=False, handlelength=0.5)
        plt.xlabel('Initial angle (deg)\n{}'.format(mode_list[i]))
        if i == 0:
            plt.ylabel(r'Mean of angle (deg)')
    fig.tight_layout()
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_thigh_angle(all_joint_data, all_condition, phase_name='thigh_angle', plot_mean=False):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 3))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    for i in range(len(mode_list)):
        plt.subplot(1, len(mode_list), i + 1)
        for j in range(len(np.unique(all_condition[i]))):
            if plot_mean:
                mean_joint_angle = Algo.extract_monotonous_joint_angle(
                    np.mean(all_joint_data[i][all_condition[i] == j, :, 2], axis=0))
                plt.plot(np.arange(101), mean_joint_angle, color=cm_fun(j))
            else:
                plt.plot(np.arange(101), all_joint_data[i][all_condition[i] == j, :, 2].T, color=cm_fun(j))
        plt.legend(legend_list[i], frameon=False, handlelength=0.5)
        plt.xlabel('{} phase (%)\n{}'.format(phase_name, mode_list[i]))
        if i == 0:
            plt.ylabel(r'Thigh angle (deg)')
    fig.tight_layout()
    if not os.path.exists('results'):
        os.mkdir('results')
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if plot_mean:
        image_name = 'mean_of_{}_thigh_monotonous_angle'.format(phase_name)
    else:
        image_name = '{}_thigh_monotonous_angle'.format(phase_name)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')


def plot_joint_angle(all_joint_data, all_condition, phase_name='thigh_angle', plot_mean=False):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 9))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    y_label_list = ['Thigh', 'Knee', 'Ankle']
    for i in range(len(mode_list)):
        for k in range(3):
            plt.subplot(3, len(mode_list), k * len(mode_list) + i + 1)
            for j in range(len(np.unique(all_condition[i]))):
                joint_angle_mat = all_joint_data[i][all_condition[i] == j, :, 2:5]
                if joint_angle_mat.shape[0] > 1:
                    joint_angle_mat = np.mean(joint_angle_mat, axis=0)
                elif joint_angle_mat.shape[0] == 1:
                    joint_angle_mat = joint_angle_mat[0]
                else:
                    continue
                if 'all' not in phase_name:
                    indices = Algo.calc_monotonous_joint_angle_indices(joint_angle_mat[:, 0], phase_name)
                    joint_angle_mat = Algo.fit_joint_angle_mat(indices, joint_angle_mat[indices])
                plt.plot(np.arange(101), joint_angle_mat[:, k], color=cm_fun(j))
                # plt.plot(joint_angle_mat[:, k], np.arange(101), color=cm_fun(j))
            plt.legend(legend_list[i], frameon=False, handlelength=0.5)
            if 2 == k:
                plt.xlabel('{} phase (%)\n{}'.format(phase_name, mode_list[i]))
            if i == 0:
                plt.ylabel(r'{} angle (deg)'.format(y_label_list[k]))

    fig.tight_layout()
    if not os.path.exists('results'):
        os.mkdir('results')
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    image_name = 'mean_of_{}_joint_angle'.format(phase_name)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')


def plot_initial_mean_angle(all_joint_data, all_condition, image_name='initial_mean_angle'):
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    fig = plt.figure(figsize=(16, 3))
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    for i in range(len(mode_list)):
        plt.subplot(1, len(mode_list), i + 1)
        plt.plot(np.mean(all_joint_data[i][:-1, :, 2], axis=-1), np.mean(all_joint_data[i][1:, :, 2], axis=-1), '.')
        plt.xlabel('Mean angle in the current gait (deg)'.format(mode_list[i]))
        for j in range(len(np.unique(all_condition[i]))):
            plt.plot(np.mean(all_joint_data[i][all_condition[i] == j, :5, 2], axis=-1),
                     np.mean(all_joint_data[i][all_condition[i] == j, :, 2], axis=-1), '.')
        plt.legend(legend_list[i], frameon=False, handlelength=0.5)
        if i == 0:
            plt.ylabel(r'Mean angle in the last gait (deg)')
    fig.tight_layout()
    if not os.path.exists('results'):
        os.mkdir('results')
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()


def plot_stance_swing_joint_angle(joint_angle_mean_dict, predicted_joint_angle_list=None, all_phase_list=None):
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
    ylabel_list = ['Predicted phase (%)', r'Thigh angle ($\degree$)',
                   r'Knee angle ($\degree$)', r'Ankle angle ($\degree$)']
    fig = plt.figure(figsize=(16, 12))
    for r in range(len(mode_list)):
        plt.subplot(4, len(mode_list), r + 1)
        if all_phase_list is not None:
            for c in [1]:
                plt.plot(np.arange(200), all_phase_list[r][:, c])
        if r == 0:
            plt.ylabel(ylabel_list[0])
        for k in range(3):
            plt.subplot(4, len(mode_list), (k + 1) * len(mode_list) + r + 1)
            # for c in [1]:
            for c in range(len(legend_list[r])):
                joint_angle_mean_vec = np.concatenate([joint_angle_mean_dict[phase_name_list[0]][r][:-1, k, c],
                                                       joint_angle_mean_dict[phase_name_list[1]][r][:-1, k, c]], axis=0)
                plt.plot(np.arange(200), joint_angle_mean_vec)
                if k > 0 and predicted_joint_angle_list is not None:
                    plt.plot(np.arange(200), predicted_joint_angle_list[r][:, k, c], '--')
                    plt.legend(['Actual', 'Predicted'], loc='lower left', frameon=False, handlelength=0.5)
            if r == 0:
                plt.ylabel(ylabel_list[k + 1])
        plt.xlabel('Actual phase (%)')
    fig.tight_layout()
    plt.show()

def plot_all_joint_angle_vertical(joint_angle_mean_dict, predicted_joint_angle_list=None, all_phase_list=None):
    '''
            joint_angle_mean_dict: {'stance', 'swing'}
            joint_angle_mean_dict['stance']: [mode number]
            joint_angle_mean_dict['stance'][0]: (101, joint number, types)
        '''
    mode_list = ['Walk', 'Stairascent', 'Stairdescent', 'Rampascent', 'Rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm'],
                   ['102 mm', '127 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    ylabel_list = ['Predicted phase (%)', r'Thigh angle ($\degree$)',
                   r'Knee angle ($\degree$)', r'Ankle angle ($\degree$)']
    condition_list = [1, 1, 1, 3, 3]
    joint_angle_mean_list = joint_angle_mean_dict['all']
    fig = plt.figure(figsize=(8, 10))
    cm_fun = cm.get_cmap('tab10', 10)
    for r in range(len(mode_list)):
        plt.subplot(len(mode_list), 4 , 4 * r + 2)
        if all_phase_list is not None:
            phase_legend_list = []
            for c in [condition_list[r]]:
                if (0 == r and 0 == c):
                    continue
                actual_phase = np.arange(101)
                predicted_phase = all_phase_list[r][:, c]
                plt.plot(actual_phase, predicted_phase, '--', color=cm_fun(1), linewidth=2,zorder = 2)
                plt.plot(actual_phase, actual_phase, color=cm_fun(0), linewidth=2,zorder = 1)
                r_corr, _ = pearsonr(actual_phase, predicted_phase)
                rmse = np.sqrt(Algo.mse(actual_phase, predicted_phase))
                # phase_legend_list.append('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            # plt.legend(phase_legend_list, loc='best', frameon=False, handlelength=0.5)
        if r == len(mode_list) - 1:
            plt.xlabel('Actual phase (%)'.format(mode_list[r]))
        plt.ylabel('{}\n{}'.format(mode_list[r], ylabel_list[0]))
        for k in range(3):
            if k == 0:
                plt.subplot(len(mode_list), 4, 4 * r + (k + 1))
            else:
                plt.subplot(len(mode_list), 4, 4 * r + (k + 2))
            for c in [condition_list[r]]:
                if (0 == r and 0 == c):
                    continue
                joint_angle_mean_vec = joint_angle_mean_list[r][:, k, c]
                plt.plot(np.arange(101), joint_angle_mean_vec, color=cm_fun(0), linewidth=2)
                # if k == 0:
                    # plt.legend(legend_list[r], loc='best', frameon=False, handlelength=0.5)
                if k > 0 and predicted_joint_angle_list is not None:
                    plt.plot(np.arange(101), predicted_joint_angle_list[r][:, k, c], '--', color=cm_fun(1), linewidth=2)
                    # plt.legend(['Actual', 'Predicted'], loc='best', frameon=False, handlelength=1)
                plt.ylabel(ylabel_list[k + 1])

            if r == len(mode_list)-1:
                plt.xlabel('Actual phase (%)'.format(mode_list[r]))
    fig.legend(['Predicted', 'Actual'], loc='lower center', ncol=2, bbox_to_anchor=(0.50, 0.97), frameon=False)
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'static_results_on_dataset'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()

def plot_all_joint_angle(joint_angle_mean_dict, predicted_joint_angle_list=None, all_phase_list=None):
    '''
        joint_angle_mean_dict: {'stance', 'swing'}
        joint_angle_mean_dict['stance']: [mode number]
        joint_angle_mean_dict['stance'][0]: (101, joint number, types)
    '''
    mode_list = ['walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent']
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm'],
                   ['102 mm', '127 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    ylabel_list = ['Predicted phase (%)', r'Thigh angle ($\degree$)',
                   r'Knee angle ($\degree$)', r'Ankle angle ($\degree$)']
    joint_angle_mean_list = joint_angle_mean_dict['all']
    fig = plt.figure(figsize=(16, 9))
    cm_fun = cm.get_cmap('tab10', 10)
    for r in range(len(mode_list)):
        plt.subplot(4, len(mode_list), r + 1)
        if all_phase_list is not None:
            phase_legend_list = []
            for c in range(len(legend_list[r])):
                if (0 == r and 0 == c):
                    continue
                actual_phase = np.arange(101)
                predicted_phase = all_phase_list[r][:, c]
                plt.plot(actual_phase, predicted_phase, color=cm_fun(c))
                r_corr, _ = pearsonr(actual_phase, predicted_phase)
                rmse = np.sqrt(Algo.mse(actual_phase, predicted_phase))
                phase_legend_list.append('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            plt.legend(phase_legend_list, loc='best', frameon=False, handlelength=0.5)
        if r == 0:
            plt.ylabel(ylabel_list[0])
        for k in range(3):
            plt.subplot(4, len(mode_list), (k + 1) * len(mode_list) + r + 1)
            for c in range(len(legend_list[r])):
                if (0 == r and 0 == c):
                    continue
                joint_angle_mean_vec = joint_angle_mean_list[r][:, k, c]
                plt.plot(np.arange(101), joint_angle_mean_vec, color=cm_fun(c), linewidth=1)
                if k == 0:
                    plt.legend(legend_list[r], loc='best', frameon=False, handlelength=0.5)
                if k > 0 and predicted_joint_angle_list is not None:
                    plt.plot(np.arange(101), predicted_joint_angle_list[r][:, k, c], '--', color=cm_fun(c), linewidth=2)
                    plt.legend(['Actual', 'Predicted'], loc='best', frameon=False, handlelength=1)
            if r == 0:
                plt.ylabel(ylabel_list[k + 1])
        plt.xlabel('Actual phase (%)\n{}'.format(mode_list[r]))
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'fitted_phase_and_angle'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_mean_joint_angle(joint_angle_mean_dict):
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
    phase_name_list = ['all']
    ylabel_list = ['Predicted phase (%)', r'Thigh angle ($\degree$)',
                   r'Knee angle ($\degree$)', r'Ankle angle ($\degree$)']
    fig = plt.figure(figsize=(16, 9))
    cm_fun = cm.get_cmap('tab10', 10)
    for r in range(len(mode_list)):
        for k in range(3):
            plt.subplot(3, len(mode_list), k * len(mode_list) + r + 1)
            for c in range(len(legend_list[r])):
                for i in range(len(phase_name_list)):
                    joint_angle_vec = joint_angle_mean_dict[phase_name_list[i]][r][:, k, c]
                    plt.plot(np.arange(101), joint_angle_vec, color=cm_fun(c))
            if r == 0:
                plt.ylabel(ylabel_list[k + 1])
        plt.xlabel('Actual phase (%)')
    fig.tight_layout()
    plt.show()


def test_filter():
    x_seg = np.arange(0, 100)
    x = np.r_[x_seg, x_seg[::-1], x_seg, x_seg[::-1]]
    x_filter = Algo.filter_gait_phase(x, dt=1)
    fig = plt.figure(figsize=(16, 9))
    plt.plot(x)
    plt.plot(x_filter)
    plt.xlabel('Time step')
    plt.ylabel('Phase (%)')
    plt.legend('Actual phase', 'Filtered phase')
    fig.tight_layout()
    img_dir = 'results/images'
    image_name = 'test_filter'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight')
    plt.show()

def visualize_one_plot(val_vec, r, para_name_list, n_lines=10):
    x = np.arange(-100, 100)
    norm = mpl.colors.Normalize(vmin=val_vec.min(), vmax=val_vec.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab10)
    cmap.set_array([])
    fig = plt.figure(figsize=(4, 3))
    ax = plt.axes()
    for i in range(n_lines):
        paras = [5, 0, 1, 0]
        paras[r] = val_vec[i]
        y = Algo.fun_x_to_y(x, *paras)
        ax.plot(x, y, c=cmap.to_rgba(val_vec[i]))
    clb=fig.colorbar(cmap, ticks=val_vec, orientation="horizontal")
    # clb.ax.set_title(para_name_list[r])
    # plt.xlabel(r's')
    # plt.ylabel(r'$\phi$')
    plt.xticks([-5, 5])
    plt.yticks([-6, 9])
    plt.xlim([-5, 5])
    plt.ylim([-6, 9])
    fig.tight_layout()


def visualize_hyper_paras_of_sigmoid_one_figure():
    para_name_list = ['h', 's0', 'k', 'b']
    boundList = [5, 2, 2, 5]
    for r in range(4):
        val_vec = np.r_[np.arange(-boundList[r], boundList[r], boundList[r] / 5)]
        visualize_one_plot(val_vec, r, para_name_list, n_lines=len(val_vec))
        plt.savefig('results/images/{}.pdf'.format(para_name_list[r]), bbox_inches='tight', dpi=300)
    plt.show()

    # x = np.arange(-100, 100)
    # line, = plt.plot(x, x)
    # plt.xlabel(r's')
    # plt.ylabel(r'$\phi$')
    # # plt.title('h=1')
    #
    #
    #
    # for r in range(4):
    #     plt.subplot(2, 2, r+1)
    #     paras = [5, 0, 1, 0]  # h, x0, k, b
    #     val_vec = np.r_[np.arange(-boundList[r], boundList[r], boundList[r]/5)]
    #     norm = mpl.colors.Normalize(vmin=0, vmax=len(val_vec)-1)
    #     cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.tab10)
    #     cmap.set_array([])
    #
    #     for c in range(len(val_vec)):
    #         paras[r] = val_vec[c]
    #         y = Algo.fun_x_to_y(x, *paras)
    #         plt.plot(x, y, c=cmap.to_rgba(c + 1))
    #         # line.set_ydata(y)
    #         # plt.title('{}={}'.format(para_name_list[r], val_vec[c]))
    #         # fig.canvas.draw()
    #         # fig.canvas.flush_events()
    #         # img_dir = 'results/sigmoid'
    #         # if not os.path.exists(img_dir):
    #         #     os.mkdir(img_dir)
    #         # plt.savefig('{}/{}_{:02d}.png'.format(img_dir, para_name_list[r], c), bbox_inches='tight', dpi=300)
    #     plt.title(para_name_list[r])
    #     plt.xlabel(r's')
    #     plt.ylabel(r'$\phi$')
    #     plt.xticks([-5, 5])
    #     plt.yticks([-10, 10])
    #     plt.xlim([-5, 5])
    #     plt.ylim([-10, 10])
    #     # Add color bar
    #     fig.colorbar(cmap, ticks=c)
    # plt.show()

def visualize_hyper_paras_of_sigmoid():
    x = np.arange(-100, 100)
    fig = plt.figure(figsize=(4, 4))
    para_name_list = ['h', 's0', 'k', 'b']
    line, = plt.plot(x, x)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('h=1')
    plt.xlim([-10, 10])
    plt.ylim([-15, 15])
    fig.tight_layout()
    for r in range(4):
        paras = [5, 0, 1, 0]  # h, x0, k, b
        val_vec = np.r_[np.arange(-5, 5, 0.5), np.arange(5, -5, -0.5)]
        for c in range(len(val_vec)):
            paras[r] = val_vec[c]
            y = Algo.sigmoid(x, *paras)
            line.set_ydata(y)
            plt.title('{}={}'.format(para_name_list[r], val_vec[c]))
            fig.canvas.draw()
            fig.canvas.flush_events()
            img_dir = 'results/sigmoid'
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            plt.savefig('{}/{}_{:02d}.png'.format(img_dir, para_name_list[r], c), bbox_inches='tight', dpi=300)


def generate_video_of_sigmoid():
    img_dir = 'results/sigmoid'
    for para_name in ['h', 's0', 'k', 'b']:
        img_name_vec = glob.glob('{}/{}_*.png'.format(img_dir, para_name))
        video_name = '{}_{}.mp4'.format(img_dir, para_name)
        IO.read_image_to_video(img_name_vec, video_name, fps=10)


def plot_thigh_angle_of_continuous_mode(all_joint_data, all_condition, is_subplot=True):
    continous_mode_list = [['walk-stairascent', 'stairascent', 'stairascent-walk',
                            'walk-stairdescent', 'stairdescent', 'stairdescent-walk'],
                           ['walk-rampascent', 'rampascent', 'rampascent-walk',
                            'walk-rampdescent', 'rampdescent', 'rampdescent-walk'],
                           ]
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]
    if is_subplot:
        fig = plt.figure(figsize=(16, 8))
    else:
        fig = plt.figure(figsize=(4, 2))
    condition_idx_list = [1, 3]
    ylabel_list = ['Stair height = {}'.format(legend_list[1][condition_idx_list[0]]),
                   'Ramp angle = {}'.format(legend_list[-1][condition_idx_list[1]])
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    for r in range(2):
        for c in range(len(continous_mode_list[r])):
            mode = continous_mode_list[r][c]
            condition_vec = all_condition[mode]
            joint_angle_mat = all_joint_data[mode][condition_vec == condition_idx_list[r], :,
                              2:5]  # thigh, knee, and ankle
            for i in range(2):
                _, inlier_indices = Algo.remove_outliers(joint_angle_mat[..., 0], std_ratio=1)
                joint_angle_mat = joint_angle_mat[inlier_indices]
            joint_angle_mean = np.mean(joint_angle_mat, axis=0)
            joint_angle_std = np.std(joint_angle_mat, axis=0)
            actual_phase = np.arange(101)
            predicted_phase = Algo.estimate_phase_from_thigh_angle(joint_angle_mean[:, 0])
            if is_subplot:
                plt.subplot(4, len(continous_mode_list[r]), 2 * r * len(continous_mode_list[r]) + c + 1)

                plt.plot(actual_phase, joint_angle_mean[:, 0])
                plt.fill_between(np.arange(joint_angle_mean.shape[0]),
                                 joint_angle_mean[:, 0] - joint_angle_std[:, 0],
                                 joint_angle_mean[:, 0] + joint_angle_std[:, 0],
                                 alpha=0.3, color=cm_fun(0))
                plt.title(mode)
                if 0 == c:
                    plt.ylabel('{}\n'.format(ylabel_list[r]) + r'Thigh angle $(\degree)$')
                else:
                    plt.yticks([])
                plt.ylim([-25, 55])
                plt.xticks([])
                plt.subplot(4, len(continous_mode_list[r]), (2 * r + 1) * len(continous_mode_list[r]) + c + 1)
                r_corr, _ = pearsonr(actual_phase, predicted_phase)
                rmse = np.sqrt(Algo.mse(actual_phase, predicted_phase))
                plt.plot(actual_phase, predicted_phase)
                plt.plot(actual_phase, actual_phase, '--')
                plt.legend(['Predicted', 'Actual'], loc='best', frameon=False, handlelength=1)
                plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
                plt.ylabel('Predicted phase (%)')
                if 1 == r:
                    plt.xlabel('Gait phase (%)')
                else:
                    plt.xticks([])
            else:
                plt.plot(actual_phase, joint_angle_mean[:, 0])
                plt.xlabel('Gait phase $s$ (%)')
                plt.ylabel(r'Thigh angle $\phi (\degree)$')
    fig.tight_layout()
    if not os.path.exists('results'):
        os.mkdir('results')
    img_dir = 'results/images'
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
    if is_subplot:
        image_name = 'thigh_angle_of_continuous_mode'
    else:
        image_name = 'thigh_angle_of_different_modes'
    plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
    plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()


def plot_joint_angle_of_continuous_mode_vertical(all_joint_data, all_condition):
    continous_mode_list = [['walk-stairascent', 'stairascent', 'stairascent-walk',
                            'walk-stairdescent', 'stairdescent', 'stairdescent-walk'],
                           ['walk-rampascent', 'rampascent', 'rampascent-walk',
                            'walk-rampdescent', 'rampdescent', 'rampdescent-walk'],
                           ['walk']
                           ]
    ylabel_mode_list = [['Walk-stairascent', 'Stairascent', 'Stairascent-walk',
                            'Walk-stairdescent', 'Stairdescent', 'Stairdescent-walk'],
                           ['Walk-rampascent', 'Rampascent', 'Rampascent-walk',
                            'Walk-rampdescent', 'Rampdescent', 'Rampdescent-walk'],
                           ['walk']
                           ]
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]

    condition_idx_list = [1, 3,0]
    ylabel_list = ['Stair height = {}'.format(legend_list[1][condition_idx_list[0]]),
                   'Ramp angle = {}'.format(legend_list[-1][condition_idx_list[1]])
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    img_name_list = ['stair', 'ramp','level_ground']
    for r in range(3):
        fig = plt.figure(figsize=(8, 10))
        for c in range(len(continous_mode_list[r])):
            mode = continous_mode_list[r][c]
            print(mode)
            condition_vec = all_condition[mode]
            joint_angle_mat = all_joint_data[mode][condition_vec == condition_idx_list[r], :,
                              2:5]  # thigh, knee, and ankle
            for i in range(2):
                _, inlier_indices = Algo.remove_outliers(joint_angle_mat[..., 0], std_ratio=1)
                joint_angle_mat = joint_angle_mat[inlier_indices]
            joint_angle_mean = np.mean(joint_angle_mat, axis=0)
            joint_angle_std = np.std(joint_angle_mat, axis=0)
            actual_phase = np.arange(101)
            predicted_phase = Algo.estimate_phase_from_thigh_angle(joint_angle_mean[:, 0])
            r_corr, _ = pearsonr(actual_phase, predicted_phase)
            rmse = np.sqrt(Algo.mse(actual_phase, predicted_phase))
            f = interpolate.interp1d(actual_phase, joint_angle_mean[:, 1], kind='cubic', axis=0)
            predicted_knee_angle_vec = f(predicted_phase)
            
            #cxx add
            save_para = 1
            if save_para:
                gait_paras_dict = Algo.calc_phase_parameters(joint_angle_mean[:, 0])
                f_joint_angle = interpolate.interp1d(np.arange(101), joint_angle_mean,
                                             kind='cubic', axis=0)
                gait_paras_dict['joint_angle_mean'] = joint_angle_mean
                gait_paras_dict['f_joint_angle'] = f_joint_angle
                np.save('data/paras/dataset_gait_paras_dict_{}.npy'.format(mode), gait_paras_dict)
            ##
            
            plt.subplot(len(continous_mode_list[r]), 4, 4 * c + 2)
            plt.plot(actual_phase, joint_angle_mean[:, 0])
            plt.fill_between(np.arange(joint_angle_mean.shape[0]),
                             joint_angle_mean[:, 0] - joint_angle_std[:, 0],
                             joint_angle_mean[:, 0] + joint_angle_std[:, 0],
                             alpha=0.3, color=cm_fun(0))
            # plt.title(mode)
            plt.ylabel('{}\n'.format(ylabel_mode_list[r][c]) + r'Thigh angle $(\degree)$')
            if len(continous_mode_list[r]) - 1 == c:
                plt.xlabel('Gait phase (%)')
            plt.ylim([-25, 55])
            
            plt.subplot(len(continous_mode_list[r]), 4, 4 * c + 1)

            plt.plot(actual_phase, predicted_phase,'--', color=cm_fun(1),zorder = 2)
            plt.plot(actual_phase, actual_phase, color=cm_fun(0),zorder = 1)
            # if 0 == c:
            #     plt.legend(['Predicted', 'Actual'], loc='best', frameon=False, handlelength=1)
            # plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            plt.ylabel('Predicted phase (%)')
            if len(continous_mode_list[r]) - 1 == c:
                plt.xlabel('Gait phase (%)')


            



            plt.subplot(len(continous_mode_list[r]), 4, 4 * c + 3)
            r_corr, _ = pearsonr(joint_angle_mean[:, 1], predicted_knee_angle_vec)
            rmse = np.sqrt(Algo.mse(joint_angle_mean[:, 1], predicted_knee_angle_vec))

            plt.plot(actual_phase, predicted_knee_angle_vec,'--', color=cm_fun(1),zorder = 2)
            plt.plot(actual_phase, joint_angle_mean[:, 1], color=cm_fun(0),zorder = 1)
            # plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            plt.ylabel(r'Knee angle $(\degree)$')
            if len(continous_mode_list[r]) - 1 == c:
                plt.xlabel('Gait phase (%)')

            plt.subplot(len(continous_mode_list[r]), 4, 4 * c + 4)
            f = interpolate.interp1d(actual_phase, joint_angle_mean[:, 2], kind='cubic', axis=0)
            predicted_ankle_angle_vec = f(predicted_phase)

            r_corr, _ = pearsonr(joint_angle_mean[:, 2], predicted_ankle_angle_vec)
            rmse = np.sqrt(Algo.mse(joint_angle_mean[:, 2], predicted_ankle_angle_vec))
            plt.plot(actual_phase, predicted_ankle_angle_vec, '--', color=cm_fun(1),zorder = 2)
            plt.plot(actual_phase, joint_angle_mean[:, 2], color=cm_fun(0),zorder = 1)

            # plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
            plt.ylabel(r'Ankle angle $(\degree)$')
            if len(continous_mode_list[r]) - 1 == c:
                plt.xlabel('Gait phase (%)')

        fig.legend(['Predicted', 'Actual'], loc='lower center', ncol=2, bbox_to_anchor=(0.50, 0.96), frameon=False)
        fig.tight_layout()
        if not os.path.exists('results'):
            os.mkdir('results')
        img_dir = 'results/images'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        image_name = 'transition_mode_{}'.format(img_name_list[r])
        plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
        plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
    plt.show()




def plot_joint_angle_of_continuous_mode(all_joint_data, all_condition, is_subplot=True):
    continous_mode_list = [['walk-stairascent', 'stairascent', 'stairascent-walk',
                            'walk-stairdescent', 'stairdescent', 'stairdescent-walk'],
                           ['walk-rampascent', 'rampascent', 'rampascent-walk',
                            'walk-rampdescent', 'rampdescent', 'rampdescent-walk'],
                           ]
    legend_list = [['1.45 m/s', '1.17 m/s', '0.88 m/s'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   ['102 mm', '127 mm', '152 mm', '178 mm'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   [r'5.2$\degree$', r'7.8$\degree$', r'9.2$\degree$', r'11$\degree$', r'12.4$\degree$',
                    r'18$\degree$'],
                   ]

    condition_idx_list = [1, 3]
    ylabel_list = ['Stair height = {}'.format(legend_list[1][condition_idx_list[0]]),
                   'Ramp angle = {}'.format(legend_list[-1][condition_idx_list[1]])
                   ]
    cm_fun = cm.get_cmap('tab10', 10)
    if not is_subplot:
        fig = plt.figure(figsize=(4, 2))
    for r in range(2):
        if is_subplot:
            fig = plt.figure(figsize=(16, 8))
        for c in range(len(continous_mode_list[r])):
            mode = continous_mode_list[r][c]
            condition_vec = all_condition[mode]
            joint_angle_mat = all_joint_data[mode][condition_vec == condition_idx_list[r], :,
                              2:5]  # thigh, knee, and ankle
            for i in range(2):
                _, inlier_indices = Algo.remove_outliers(joint_angle_mat[..., 0], std_ratio=1)
                joint_angle_mat = joint_angle_mat[inlier_indices]
            joint_angle_mean = np.mean(joint_angle_mat, axis=0)
            joint_angle_std = np.std(joint_angle_mat, axis=0)
            actual_phase = np.arange(101)
            predicted_phase = Algo.estimate_phase_from_thigh_angle(joint_angle_mean[:, 0])
            r_corr, _ = pearsonr(actual_phase, predicted_phase)
            rmse = np.sqrt(Algo.mse(actual_phase, predicted_phase))
            f = interpolate.interp1d(actual_phase, joint_angle_mean[:, 1], kind='cubic', axis=0)
            predicted_knee_angle_vec = f(predicted_phase)
            if is_subplot:
                plt.subplot(4, len(continous_mode_list[r]), c + 1)
                plt.plot(actual_phase, joint_angle_mean[:, 0])
                plt.fill_between(np.arange(joint_angle_mean.shape[0]),
                                 joint_angle_mean[:, 0] - joint_angle_std[:, 0],
                                 joint_angle_mean[:, 0] + joint_angle_std[:, 0],
                                 alpha=0.3, color=cm_fun(0))
                plt.title(mode)
                if 0 == c:
                    plt.ylabel('{}\n'.format(ylabel_list[r]) + r'Thigh angle $(\degree)$')
                else:
                    plt.yticks([])
                plt.ylim([-25, 55])
                plt.xticks([])

                plt.subplot(4, len(continous_mode_list[r]), 1 * len(continous_mode_list[r]) + c + 1)

                plt.plot(actual_phase, predicted_phase, color=cm_fun(1))
                plt.plot(actual_phase, actual_phase, '--', color=cm_fun(0))
                if 0 == c:
                    plt.legend(['Predicted', 'Actual'], loc='best', frameon=False, handlelength=1)
                plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
                plt.ylabel('Predicted phase (%)')

                plt.subplot(4, len(continous_mode_list[r]), 2 * len(continous_mode_list[r]) + c + 1)

                r_corr, _ = pearsonr(joint_angle_mean[:, 1], predicted_knee_angle_vec)
                rmse = np.sqrt(Algo.mse(joint_angle_mean[:, 1], predicted_knee_angle_vec))

                plt.plot(actual_phase, predicted_knee_angle_vec, color=cm_fun(1))
                plt.plot(actual_phase, joint_angle_mean[:, 1], '--', color=cm_fun(0))
                plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
                plt.ylabel(r'Knee angle $(\degree)$')

                plt.subplot(4, len(continous_mode_list[r]), 3 * len(continous_mode_list[r]) + c + 1)
                f = interpolate.interp1d(actual_phase, joint_angle_mean[:, 2], kind='cubic', axis=0)
                predicted_ankle_angle_vec = f(predicted_phase)

                r_corr, _ = pearsonr(joint_angle_mean[:, 2], predicted_ankle_angle_vec)
                rmse = np.sqrt(Algo.mse(joint_angle_mean[:, 2], predicted_ankle_angle_vec))
                plt.plot(actual_phase, predicted_ankle_angle_vec, color=cm_fun(1))
                plt.plot(actual_phase, joint_angle_mean[:, 2], '--', color=cm_fun(0))

                plt.title('r = {:.3f}, rmse = {:.3f}'.format(r_corr, rmse))
                plt.ylabel(r'Ankle angle $(\degree)$')

                if 1 == r:
                    plt.xlabel('Gait phase (%)')
                else:
                    plt.xticks([])
            else:
                plt.plot(actual_phase, predicted_knee_angle_vec)
                plt.xlabel('Gait phase $s$ (%)')
                plt.ylabel(r'Knee angle $\theta (\degree)$')

        fig.tight_layout()
        if is_subplot or 1 == r:
            if not os.path.exists('results'):
                os.mkdir('results')
            img_dir = 'results/images'
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            if is_subplot:
                image_name = 'joint_angle_of_{}'.format(continous_mode_list[r][0])
            else:
                image_name = 'fitted_angle_of_{}'.format(continous_mode_list[r][0])
            plt.savefig('{}/{}.pdf'.format(img_dir, image_name), bbox_inches='tight')
            plt.savefig('{}/{}.png'.format(img_dir, image_name), bbox_inches='tight', dpi=300)
            plt.show()


def phase_estimation_init_plot(phase_vec_list=None, joint_angle_vec_list=None):
    '''
    phase_vec_list: [predicted, actual]
    joint_angle_list: [predicted, actual]
    '''
    if phase_vec_list is None:
        phase_vec_list = [np.arange(101), np.arange(101)]
    if joint_angle_vec_list is None:
        joint_angle_vec_list = [np.zeros((101, 3)), np.zeros((101, 3))]
    fig = plt.figure(figsize=(16, 4))
    line_list = []
    line_type_list = ['-', '--']
    cm_fun = cm.get_cmap('tab10', 10)
    plt.subplot(1, 3, 1)
    for r in range(2):
        joint_angle_vec = joint_angle_vec_list[r]
        joint_x_vec, joint_y_vec = Algo.calc_joint_points(joint_angle_vec[-1])
        line, = plt.plot(joint_x_vec, joint_y_vec, line_type_list[r], marker='o',
                         linewidth=5, markersize=10, color=cm_fun(r))
        line_list.append(line)
    plt.legend(['Predicted', 'Actual'])
    plt.ylim([-3, 1])
    plt.xlim([-3, 3])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.subplot(1, 3, 2)
    for r in range(2):
        joint_angle_vec = joint_angle_vec_list[r]
        for i in range(joint_angle_vec.shape[1]):
            line, = plt.plot(joint_angle_vec[:, i], line_type_list[r], color=cm_fun(2 * i + r))
            line_list.append(line)
    plt.ylim([-90, 90])
    plt.xlabel('Time step')
    plt.ylabel(r'Angle $(\degree)$')
    plt.legend(['Thigh', 'Knee', 'Ankle'])

    plt.subplot(1, 3, 3)
    for r in range(2):
        phase_vec = phase_vec_list[r]
        line, = plt.plot(phase_vec, line_type_list[r], color=cm_fun(r))
        line_list.append(line)
    plt.ylim([-1, 101])
    plt.xlabel('Time step')
    plt.ylabel(r'Gait phase (%)')
    plt.legend(['Predicted', 'Actual'])
    fig.tight_layout()
    plt.pause(0.1)
    return fig, line_list


def phase_estimation_update_plot(fig, line_list, phase_vec_list, joint_angle_vec_list, idx, is_savefig=True):
    '''
    joint_angle_vec_list: in degree
    '''
    for r in range(2):
        joint_angle_vec = joint_angle_vec_list[r]
        joint_x_vec, joint_y_vec = Algo.calc_joint_points(np.deg2rad(joint_angle_vec[-1]))
        line_list[r].set_xdata(joint_x_vec)
        line_list[r].set_ydata(joint_y_vec)
        for i in range(joint_angle_vec.shape[1]):
            line_list[3 * r + i + 2].set_ydata(joint_angle_vec[:, i])
        line_list[r + 8].set_ydata(phase_vec_list[r])
    fig.canvas.draw()
    fig.canvas.flush_events()
    if is_savefig:
        img_dir = 'results/imu_offline'
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)
        plt.savefig('{}/{:04d}.png'.format(img_dir, idx), bbox_inches='tight')


def IMUReader_init_plot():
    fig = plt.figure(figsize=(16, 4))
    roll_vec = np.zeros((100, 3))
    acc_z_vec = np.zeros((100, 3))
    joint_angle_vec, joint_x_vec, joint_y_vec = Algo.calc_leg_data(roll_vec)
    line_list = []

    plt.subplot(1, 3, 1)
    line, = plt.plot(joint_x_vec, joint_y_vec, marker='o', linewidth=5, markersize=10)
    line_list.append(line)
    plt.ylim([-3, 1])
    plt.xlim([-3, 3])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    plt.subplot(1, 3, 2)
    for i in range(joint_angle_vec.shape[1]):
        line, = plt.plot(joint_angle_vec[:, i])
        line_list.append(line)
    plt.ylim([-4, 4])
    plt.xlabel('Time step')
    plt.ylabel(r'Angle $(\degree)$')
    plt.legend(['Thigh', 'Knee', 'Ankle'])

    plt.subplot(1, 3, 3)
    for i in range(acc_z_vec.shape[1]):
        line, = plt.plot(acc_z_vec[:, i])
        line_list.append(line)
    plt.ylim([-10, 10])
    plt.xlabel('Time step')
    plt.ylabel(r'Acceleration $(m/s^{2})$')

    fig.tight_layout()
    plt.pause(0.1)
    return fig, line_list


def IMUReader_update_plot(fig, line_list, roll_vec, acc_z_vec):
    joint_angle_vec, joint_x_vec, joint_y_vec = Algo.calc_leg_data(roll_vec)
    line_list[0].set_xdata(joint_x_vec)
    line_list[0].set_ydata(joint_y_vec)
    for i in range(3):
        line_list[i + 1].set_ydata(joint_angle_vec[:, i])
    for i in range(3):
        line_list[i + 4].set_ydata(acc_z_vec[:, i])
    fig.canvas.draw()
    fig.canvas.flush_events()
