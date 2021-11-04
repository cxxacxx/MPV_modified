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


def view_trajectory(i, data_d_mat, data_mat):
    fig = plt.figure(figsize=(4, 8))
    y_label_list = ['Gait phase %', 'Thigh_angle (deg)', 'Knee angle (deg)', 'Ankle angle (deg)']
    idx_vec = np.linspace((i-100), i+100, num=201).astype(np.int)
    line_list = []
    y_min_vec = np.min(np.r_[data_d_mat, data_mat], axis=0)
    y_max_vec = np.max(np.r_[data_d_mat, data_mat], axis=0)
    for k in range(4):
        plt.subplot(4, 1, k + 1)
        line, = plt.plot(data_d_mat[idx_vec, k])
        line_list.append(line)
        if k >= 2:
            line, = plt.plot(data_mat[idx_vec, k], '--')
            line_list.append(line)
            plt.legend(['Command', 'Actual'])
        if k == 3:
            plt.xlabel('Time step')
        plt.ylabel(y_label_list[k])
        plt.ylim(y_min_vec[k]-1, y_max_vec[k]+1)
    plt.tight_layout()
    # plt.savefig('results/walking_test/joint_angle_{}.png'.format(i), dpi=300)
    return fig, line_list


def update_plot(fig, line_list, i, data_d_mat, data_mat):
    idx_vec = np.linspace((i - 100), i + 100, num=201).astype(np.int)
    for k in range(2):
        line_list[k].set_ydata(data_d_mat[idx_vec, k])
    for j in range(2):
        line_list[2*j+2].set_ydata(data_d_mat[idx_vec, j+2])
        line_list[2 * j + 3].set_ydata(data_mat[idx_vec, j + 2])
    fig.canvas.draw()
    # plt.pause(1e-3)
    plt.savefig('results/walking_test/joint_angle_{}.png'.format(i), dpi=150)


def test_offline_imu_reader():
    data_mat = np.load('data/walking.npy')
    init_idx = np.where(data_mat[:, 0] != 0)[0][0]
    data_mat = data_mat[init_idx + 500:-100]  # the 350 is determined by observing the whole data
    phase_vec_list, joint_angle_vec_list = Algo.analyze_offline_imu_data(data_mat)

def analyze_transition(file_name='data/processed/joint_data_of_continuous_mode.npy'):
    joint_data = np.load(file_name, allow_pickle=True).item()
    phase_joint_data = joint_data['phase_joint_data']
    phase_condition = joint_data['phase_condition']
    Plot.plot_joint_angle_of_continuous_mode_vertical(phase_joint_data, phase_condition)


def main():
    # test_offline_imu_reader()
    '''Fig 2'''
    # Plot.visualize_hyper_paras_of_sigmoid_one_figure() # figure 2
    '''Fig 3'''
    #simulation.compare_different_phase_variables(is_video=False) # figure 3
    '''Fig 4'''
    # plot_data = np.load('data/processed/plot_data.npy', allow_pickle=True).item()
    # Plot.plot_all_joint_angle_vertical(plot_data['joint_angle_mean_dict'], plot_data['predicted_joint_angle_list'],
    #                             plot_data['all_phase_list'])
    '''Fig 5'''
    analyze_transition()

    #Plot.plot_joint_angle_and_cubit_fitting_one_figure()

    # data_dict = np.load('results/walking_test/walking_test.npy', allow_pickle=True).item()
    # q_qv_d_mat = data_dict['q_qv_d_mat']
    # data_d_mat = np.zeros((q_qv_d_mat.shape[0], 4))
    # data_mat = np.zeros((q_qv_d_mat.shape[0], 4))
    # data_d_mat[:, 0] = data_dict['phase_save_vec']
    # data_d_mat[:, 1] = data_dict['q_thigh_vec']
    # data_d_mat[:, 2:] = data_dict['q_qv_d_mat'][:, ::2]
    # data_mat[:, 2:] = data_dict['q_qv_mat'][:, ::2]
    # Algo.analyze_offline_imu_data(data_mat)
    # #plot trajectories
    # fig, line_list = view_trajectory(2500, data_d_mat, data_mat)
    # for i in tqdm(np.arange(2500, len(data_d_mat)-100, 5)):
    #     update_plot(fig, line_list, i, data_d_mat, data_mat)
    # segment_gait(data_d_mat,data_mat)


def segment_gait(data_d_mat, data_mat):
    data_d_mat = np.c_[data_d_mat, data_mat[:, 2:]]
    phase_vec = data_d_mat[:, 0]
    phase_vec_diff = phase_vec[1:] - phase_vec[:-1]
    gait_event_indices = np.where(phase_vec_diff<-80)[0]+1
    steps = len(gait_event_indices)
    init_step = 10
    data_in_gait_mat = np.zeros((steps-init_step, 101, 6))
    for i in range(steps-init_step):
        step_indices = np.arange(gait_event_indices[i+init_step-1], gait_event_indices[i + init_step])
        data_mat_i = data_d_mat[step_indices]
        data_in_gait_mat[i] = Algo.interpolate_joint_angle(data_mat_i, step_indices)
        # plt.plot(data_mat_i[:, 0])
        # plt.plot(data_in_gait_mat[i,:, 0], '--')


    data_in_gait_mean = np.mean(data_in_gait_mat, axis=0)
    data_in_gait_std = np.std(data_in_gait_mat, axis=0)
    cm_fun = cm.get_cmap('tab10', 10)
    desired_data_mat = np.zeros((101, 4))
    desired_phase_vec = np.linspace(0, 100, 101)
    desired_data_mat[:, 0] = desired_phase_vec
    gait_paras_dict, f_joint_angle = load_gait_paras()
    desired_data_mat[:, 1:] = f_joint_angle(desired_phase_vec)
    fig = plt.figure(figsize=(8, 8))
    y_label_list = ['Gait phase %', 'Thigh_angle (deg)', 'Knee angle (deg)', 'Ankle angle (deg)']
    for i in [3, 2, 1, 0]:
        plt.subplot(2, 2, i+1)
        plt.plot(desired_phase_vec, desired_data_mat[:, i], '--')
        plt.plot(desired_phase_vec, data_in_gait_mean[:, i])
        if i > 1:
            plt.plot(desired_phase_vec, data_in_gait_mean[:, i+2])
        rmse = Algo.rmse(desired_data_mat[:, i], data_in_gait_mean[:, i], axis=0)
        r, _ = pearsonr(desired_data_mat[:, i], data_in_gait_mean[:, i])
        plt.xlabel('Gait phase (%)')
        plt.ylabel(y_label_list[i])
        plt.title('RMSE: {:.3f}, R: {:.3f}'.format(rmse, r))
    for i in [3, 2, 1, 0]:
        plt.subplot(2, 2, i + 1)
        plt.fill_between(desired_phase_vec,
                         data_in_gait_mean[:, i] - data_in_gait_std[:, i],
                         data_in_gait_mean[:, i] + data_in_gait_std[:, i],
                         alpha=0.3, color=cm_fun(1))
        if i > 1:
            plt.fill_between(desired_phase_vec,
                            data_in_gait_mean[:, i+2] + data_in_gait_std[:, i+2],
                            data_in_gait_mean[:, i+2] - data_in_gait_std[:, i+2],
                            alpha=0.3, color=cm_fun(2))
    fig.legend(['Desired', 'Command', 'Measured'], loc='lower center',
               ncol=3, bbox_to_anchor=(0.50, 0.96), frameon=False)
    fig.tight_layout()
    plt.savefig('results/walking_test/walking_test_results.png', dpi=300)
    plt.show()




def generate_video():
    img_name_vec = glob.glob('results/walking_test/*.png')
    sorted(img_name_vec)
    video_name = 'results/video/walking_test_data.mp4'
    IO.read_image_to_video(img_name_vec, video_name, fps=20)



def load_gait_paras(mode='level_ground'):
    '''
    Available modes: 'level_ground', 'stair_ascent', 'stair_descent', 'ramp_ascent', 'ramp_descent'
    '''
    gait_paras_dict = np.load('data/paras/gait_paras_dict_{}.npy'.format(mode), allow_pickle=True).item()
    f_joint_angle = gait_paras_dict['f_joint_angle']
    return gait_paras_dict, f_joint_angle

def visualize_benchtop_test():
    bentop_test_data = np.load('results/benchtop_test/bentop_test.npy', allow_pickle=True).item()
    data_mat = bentop_test_data['data_mat']
    data_mat_new = bentop_test_data['data_mat_new']
    y_label_list = ['Knee angle (deg)', 'Knee angular velocity (deg/s)', 'Ankle angle (deg)',
                    'Ankle angular velocity (deg/s)']
    fig = plt.figure(figsize=(8, 8))
    time_step = 1e-2 # 10 ms
    b_float_2_int = 30000
    k_float_2_int = 100
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        t_vec = np.linspace(0, time_step * len(data_mat[:, i]), len(data_mat[:, i]))
        command_vec = (data_mat[:, i].astype(float) - b_float_2_int) / k_float_2_int
        measured_vec = (data_mat_new[:, i].astype(float) - b_float_2_int) / k_float_2_int
        rmse = Algo.rmse(command_vec, measured_vec, axis=0)
        r, _ = pearsonr(command_vec, measured_vec)
        plt.title('RMSE: {:.3f}, R: {:.3f}'.format(rmse, r))
        plt.plot(t_vec, command_vec, '--')
        plt.plot(t_vec, measured_vec)
        plt.xlabel('Time (s)')
        plt.ylabel(y_label_list[i])
    fig.legend(['Command', 'Measured'], loc='lower center',
               ncol=2, bbox_to_anchor=(0.50, 0.96), frameon=False)
    plt.tight_layout()
    plt.savefig('results/benchtop_test/measured.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    # visualize_benchtop_test()
    # generate_video()
    main()