import numpy as np
import matplotlib
import time
from Utils import Algo, IO, Plot

#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import glob

is_ubuntu = True
if not is_ubuntu:
    plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})


def main():
    if is_ubuntu:
        joint_angle_mean_dict = IO.read_mean_joint_angle()
        all_phase_list = Algo.estimate_all_phase(joint_angle_mean_dict)
        predicted_joint_angle_list = Algo.estimate_all_angle(joint_angle_mean_dict, all_phase_list)
        plot_data = {'joint_angle_mean_dict': joint_angle_mean_dict,
                     'predicted_joint_angle_list': predicted_joint_angle_list,
                     'all_phase_list': all_phase_list}
        np.save('data/processed/plot_data.npy', plot_data)
    else:
        plot_data = np.load('data/processed/plot_data.npy', allow_pickle=True).item()
    Plot.plot_all_joint_angle(plot_data['joint_angle_mean_dict'], plot_data['predicted_joint_angle_list'],
                              plot_data['all_phase_list'])


def analyze_transition(is_ubuntu=False, file_name='data/processed/joint_data_of_continuous_mode.npy'):
    if is_ubuntu:
        gait_data = IO.read_joint_angles_with_gait_phases(read_csv=False)
        phase_joint_data, phase_condition = Algo.segment_gait_to_phases(
            gait_data, phase_name='all', leg_idx=0, ref_leg_idx=0, remove_outlier=False)
        np.save(file_name, {'phase_joint_data': phase_joint_data, 'phase_condition': phase_condition})
    else:
        joint_data = np.load(file_name, allow_pickle=True).item()
        phase_joint_data = joint_data['phase_joint_data']
        phase_condition = joint_data['phase_condition']
    Plot.plot_joint_angle_of_continuous_mode_vertical(phase_joint_data, phase_condition)
    # Plot.plot_thigh_angle_of_continuous_mode(phase_joint_data, phase_condition, is_subplot=False)




def test_offline_imu_reader():
    data_mat = np.load('data/walking.npy')
    init_idx = np.where(data_mat[:, 0] != 0)[0][0]
    data_mat = data_mat[init_idx + 500:-100]  # the 350 is determined by observing the whole data
    phase_vec_list, joint_angle_vec_list = Algo.analyze_offline_imu_data(data_mat)
    fig, line_list = Plot.phase_estimation_init_plot()
    for i in range(phase_vec_list[0].shape[0] - 100):
        Plot.phase_estimation_update_plot(
            fig, line_list, [phase_vec_list[0][i:i + 101], phase_vec_list[1][i:i + 101]],
            [joint_angle_vec_list[0][i:i + 101], joint_angle_vec_list[1][i:i + 101]], i, is_savefig=(0 == i % 10))


def test_online_imu_reader():
    data_mat = np.load('data/walking.npy')
    init_idx = np.where(data_mat[:, 0] != 0)[0][0]
    data_mat = data_mat[init_idx + 500:-100]  # the 350 is determined by observing the whole data
    Algo.analyze_online_imu_data(data_mat)


def capture_imu_and_estimate_phase():
    original_data_mat = np.load('data/walking.npy')
    init_idx = np.where(original_data_mat[:, 0] != 0)[0][0]
    original_data_mat = original_data_mat[init_idx + 500:-100]  # the 500 is determined by observing the whole data
    data_mat = np.zeros((original_data_mat.shape[0], 3 * 9 + 5))
    time_vec = original_data_mat[..., -1]  # time
    predictor = Algo.OnlineGaitPredicter()
    time_init = time.time()
    for i in range(1, original_data_mat.shape[0]):
        captured_imu_data_vec = original_data_mat[i]
        predicted_results = predictor.forward(captured_imu_data_vec, dt=100 * (time_vec[i] - time_vec[i - 1]))
        if predicted_results is None:
            continue
        else:
            phase, estimated_joint_angle = predicted_results
            data_vec = np.zeros(data_mat.shape[-1])
            data_vec[:-4] = captured_imu_data_vec
            data_vec[-4] = phase
            data_vec[-3:] = estimated_joint_angle
            Algo.fifo_data_vec(data_mat, data_vec)
            print('Costed time: {:.3f} s'.format(time.time() - time_init))
            time_init = time.time()
            time.sleep(8e-3)
        if 0 == i % 10:
            np.save('data/estimated_walking.npy', data_mat)


if __name__ == '__main__':
    #test_online_imu_reader()
    analyze_transition(is_ubuntu=False)
    #capture_imu_and_estimate_phase()
    #test_offline_imu_reader()
    # IO.read_image_to_video(glob.glob('results/imu_offline/*.png'), 'results/video/imu_offline.mp4', fps=5)
    #Algo.analyze_offline_outdoor_data(signal_folder='data/IMUOutEvening')

