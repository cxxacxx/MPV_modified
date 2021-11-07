import numpy as np
import time
import serial
import matplotlib.pyplot as plt
from Utils import Algo, IO

def main():
    
    control_motor = 0
    '''Initialize state vector'''
    
    #view_joint_angle(f_joint_angle)
    
    gait_data = IO.read_separate_gait_files(read_csv=False, data_file_path='data/separate_gait_data.npy')
    gait_mode_list = gait_data['gait_mode']
    joint_data_list = gait_data['joint_angle_and_velocity']
    foot_acc_list = gait_data['foot_acc_mat']

    
    for trail in range(31,32):#range(144):
        '''Main loop'''
        trail = 66
        joint_data = joint_data_list[trail][:,3:6] # thigh, knee, and ankle
        thigh_angle = joint_data[:,0]
        foot_acc_z = foot_acc_list[trail][:,2]*9.8
        gait_mode = gait_mode_list[trail]
        
        plt.plot(thigh_angle)
        plt.plot(foot_acc_z)
        plt.show()
        time_step_num = len(foot_acc_z)
        time_vec = np.array([-0.02, -0.01])
        time_step_num = 1500
        size = int(time_step_num/2)
        q_qv_d_mat = np.zeros((size, 4))
        q_qv_mat = np.zeros((size, 4))
        q_thigh_vec = np.zeros(size)
        phase_save_vec = np.zeros(size)
        phase_d_vec = np.array([100, 100])
        q_d_mat = np.zeros((2, 2))
        gait_paras_dict, f_joint_angle = load_gait_paras('walk')
        q_d_mat[-1] = predict_q_d(f_joint_angle, phase=0)
        phase_predictor = Algo.OnlinePhasePredicter()
        
        gait_num = 0
        
        mode = 'walk'
 
        
        for time_step in range(0,time_step_num,2):
            # try:
            mode_previous = mode
            if time_step == 542:
                print("debug pause")
            mode = gait_mode[time_step]
            
            t = time_step * 0.01;               
            q_thigh = joint_data[time_step][0]
            acc_z = foot_acc_z[time_step]
            q_thigh_vec = fifo_mat(q_thigh_vec, q_thigh)
            time_vec = fifo_mat(time_vec, time_step*0.01)
            if (mode == 'idle'):
                mode = 'walk'
            if(mode!=mode_previous):
                gait_paras_dict, f_joint_angle = load_gait_paras(mode)
                dic = phase_predictor.gait_paras_dict
                if 'stance_end_idx' in dic.keys():
                    prev_stance_end_idx = dic['stance_end_idx']
                    gait_paras_dict['stance_end_idx'] = prev_stance_end_idx
                phase_predictor.gait_paras_dict = gait_paras_dict

            # dic = phase_predictor.gait_paras_dict
            # dic['stance_end_idx']
            phase = phase_predictor.forward(q_thigh, acc_z, dt = 1)
            print(time_step)
            print(phase)
            
            if phase is None:
                phase_save_vec = fifo_mat(phase_save_vec, 200)
            
            if phase is not None:
                q_qv_d, q_d_mat = predict_q_qv_d(phase, f_joint_angle, q_d_mat, time_vec)
                phase_save_vec = fifo_mat(phase_save_vec, phase)
                q_qv_d_mat = fifo_mat(q_qv_d_mat, q_qv_d)
                if control_motor:
                    send_signals_to_matlab(q_qv_d)
                time.sleep(8e-3)  # sleep 8 ms
                if control_motor:
                    q_qv = read_signals_from_motor(ser)
                    q_qv_mat = fifo_mat(q_qv_mat, q_qv)
                    print('error: {}, desired: {}; actual: {}'.format(q_qv - q_qv_d, q_qv_d, q_qv))
            else:
                time.sleep(8e-3)  # sleep 8 ms
            if time_step % 100 == 0:
                data = {'q_qv_d_mat': q_qv_d_mat, 'q_qv_mat': q_qv_mat,
                        'phase_save_vec': phase_save_vec,
                        'q_thigh_vec': q_thigh_vec}
                #np.save('results/benchtop_test_{}.npy'.format(time.time()), data)

    view_trajectory(q_qv_d_mat, q_qv_mat, phase_save_vec, q_thigh_vec)


def predict_phase(thigh_angle, phase_vec, thigh_angle_vec, state, gait_paras_dict):
    thigh_angle_vec = np.append(thigh_angle_vec, thigh_angle)
    phase, state, gait_paras_dict = Algo.estimate_phase_at_one_step(thigh_angle_vec, state, gait_paras_dict)
    phase_vec = np.append(phase_vec, phase)
    filter_phase_vec = Algo.filter_gait_phase(phase_vec, dt=1)
    filter_phase_vec[np.isnan(phase_vec)] = 100
    filter_phase_vec = np.clip(filter_phase_vec, a_min=0, a_max=100)
    phase = filter_phase_vec[-1]
    return phase, phase_vec, thigh_angle_vec, state, gait_paras_dict


def predict_q_d(f_joint_angle, phase):
    joint_angles = f_joint_angle(phase)  # hip, knee, ankle
    return joint_angles[1:]


def predict_q_qv_d(phase, f_joint_angle, q_d_mat, time_vec):
    q_d_mat = fifo_mat(q_d_mat, predict_q_d(f_joint_angle, phase))
    q_qv_d = np.zeros(4)
    q_qv_d[::2] = q_d_mat[-1]
    q_qv_d[1::2] = (q_d_mat[-1] - q_d_mat[0]) / (time_vec[-1] - time_vec[0])
    # q_qv_d[1::2] = 0
    print(phase, q_qv_d)
    return q_qv_d, q_d_mat




def read_q_acc(imu_data_buf, q_thigh_init):
    q_thigh = -(imu_data_buf[1] - q_thigh_init)
    acc = imu_data_buf[4]
    return q_thigh, acc


def initialize_q_thigh(imu_data_buf):
    q_thigh_vec = np.zeros(10)
    for i in range(10):
        q_thigh_vec[i] = imu_data_buf[1]
        time.sleep(1e-2)
    q_thigh_init = np.mean(q_thigh_vec)
    print('Initialized q_thigh!')
    return q_thigh_init



def load_gait_paras(mode='walk'):
    '''
    Available modes: 'walk', 'stairascent', 'stairdescent', 'rampascent', 'rampdescent','walk-stairascent',
    'walk-stairdescent','walk-rampascent','walk-rampdescent','stairascent-walk','stairdescent-walk',
    'rampascent-walk','rampdescent-walk'
    '''
    gait_paras_dict = np.load('data/paras/dataset_gait_paras_dict_{}.npy'.format(mode), allow_pickle=True).item()
    f_joint_angle = gait_paras_dict['f_joint_angle']
    return gait_paras_dict, f_joint_angle


def view_joint_angle(f_joint_angle):
    #plt.plot(f_joint_angle(np.linspace(0, 100, 101)) - f_joint_angle(0))
    plt.plot(f_joint_angle(np.linspace(0, 100, 101)))
    plt.legend(['Hip', 'Knee', 'Ankle'])
    plt.show()


def view_trajectory(q_qv_d_mat, q_qv_mat, phase_save_vec, q_thigh_vec):
    time_step = 1e-2
    y_label_list = ['Knee angle (deg)', 'Knee angular velocity (deg/s)', 'Ankle angle (deg)',
                    'Ankle angular velocity (deg/s)']
    fig = plt.figure()
    t_vec = np.linspace(0, time_step * len(q_qv_d_mat[:, 0]), len(q_qv_d_mat[:, 0]))
    for i in range(4):
        plt.subplot(3, 2, i + 1)
        plt.plot(t_vec, q_qv_d_mat[:, i])
        plt.plot(t_vec, q_qv_mat[:, i], '--')
        plt.xlabel('Time (s)')
        plt.ylabel(y_label_list[i])
        plt.legend(['Desired', 'Actual'])
    plt.subplot(3, 2, 5)
    plt.plot(t_vec, phase_save_vec)
    plt.ylabel('Gait phase %')
    plt.xlabel('Time (s)')
    plt.subplot(3, 2, 6)
    plt.plot(t_vec, q_thigh_vec)
    plt.ylabel('Thigh angle (deg)')
    plt.xlabel('Time (s)')
    # plt.tight_layout()
    plt.savefig('results/images/joint_angle.jpg')
    plt.savefig('results/images/joint_angle.pdf')
    data = {'q_qv_d_mat':q_qv_d_mat, 'q_qv_mat':q_qv_mat, 'phase_save_vec':phase_save_vec,
            'q_thigh_vec':q_thigh_vec}
    np.save('results/benchtop_test.npy', data)
    plt.show()


def fifo_mat(data_mat, data_vec):
    data_mat[:-1] = data_mat[1:]
    data_mat[-1] = data_vec
    return data_mat


if __name__ == '__main__':
    main()