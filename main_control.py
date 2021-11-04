import numpy as np
import time
import serial
import matplotlib.pyplot as plt
from Utils import Algo

def main(control_motor=False, is_simu = True):
    '''Initialize state vector'''
    gait_paras_dict, f_joint_angle = load_gait_paras()
    view_joint_angle(f_joint_angle)
    if not is_simu:
        imu_data_buf = np.memmap('log/imu_euler_acc.npy', dtype='float32', mode='r', shape=(6,))
        q_thigh_init = initialize_q_thigh(imu_data_buf)
    q_d_mat = np.zeros((2, 2))
    q_d_mat[-1] = predict_q_d(f_joint_angle, phase=0)
    time_vec = np.array([-2e-2, -1e-2])
    time_step_num = 1000
    q_qv_d_mat = np.zeros((time_step_num, 4))
    q_qv_mat = np.zeros((time_step_num, 4))
    q_thigh_vec = np.zeros(time_step_num)
    phase_save_vec = np.zeros(time_step_num)
    phase_d_vec = np.array([100, 100])
    phase_predictor = Algo.OnlinePhasePredicter()
    gait_num = 0
    '''Main loop'''
    if control_motor:
        ser = serial.Serial(
            port='/dev/ttyUSB1',
            baudrate=115200,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS, timeout=1e-2)
        # initialize_prosthesis(ser, q_d_mat)
        time.sleep(1e-1)  # sleep 100 ms
    start_time = time.time()

    for i in range(time_step_num):
        # try:
        t = time.time() - start_time
        if is_simu:
            gait_cycle = 2  # s
            phase_d = (100 / gait_cycle) * (t % gait_cycle)
            phase_d_vec = fifo_mat(phase_d_vec, phase_d)
            if phase_d_vec[-1] - phase_d_vec[0] < 0:
                acc_z = 20
                gait_num += 1
            else:
                acc_z = 0
            q_thigh = f_joint_angle(phase_d)[0]
        else:
            q_thigh, acc_z = read_q_acc(imu_data_buf, q_thigh_init)
        q_thigh_vec = fifo_mat(q_thigh_vec, q_thigh)
        time_vec = fifo_mat(time_vec, time.time() - start_time)
        phase = phase_predictor.forward(q_thigh, acc_z, dt = 100 * (time_vec[-1] - time_vec[0]))
        print(phase)
        if phase is not None:
            q_qv_d, q_d_mat = predict_q_qv_d(phase, f_joint_angle, q_d_mat, time_vec)
            phase_save_vec = fifo_mat(phase_save_vec, phase)
            q_qv_d_mat = fifo_mat(q_qv_d_mat, q_qv_d)
            if control_motor:
                send_signals_to_motor(ser, q_qv_d)
            time.sleep(8e-3)  # sleep 8 ms
            if control_motor:
                q_qv = read_signals_from_motor(ser)
                q_qv_mat = fifo_mat(q_qv_mat, q_qv)
                print('error: {}, desired: {}; actual: {}'.format(q_qv - q_qv_d, q_qv_d, q_qv))
        else:
            time.sleep(8e-3)  # sleep 8 ms
        if i % 100 == 0:
            data = {'q_qv_d_mat': q_qv_d_mat, 'q_qv_mat': q_qv_mat,
                    'phase_save_vec': phase_save_vec,
                    'q_thigh_vec': q_thigh_vec}
            np.save('results/benchtop_test_{}.npy'.format(time.time()), data)
        # except:
        #     print('-----------Main loop error!-------------')
    if control_motor:
        ser.close()
    view_trajectory(q_qv_d_mat, q_qv_mat, phase_save_vec, q_thigh_vec)

def detect_heel_strike():
    phase_vec = np.zeros(0)
    thigh_angle_vec = np.zeros(0)
    state = 0
    return phase_vec, thigh_angle_vec, state


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


def send_signals_to_motor(ser, q_qv_d):
    q_qv_d[1::2] = np.clip(q_qv_d[1::2], a_min=-500, a_max=500)
    byte_vec = bytearray(q_qv_to_motor_signal(q_qv_d))
    ser.write(byte_vec)


def read_signals_from_motor(ser, data_byte_size=8):
    read_byte_vec = ser.read(data_byte_size)
    q_qv_motor = np.frombuffer(read_byte_vec, dtype=np.uint16)
    if len(q_qv_motor) == 4:
        q_qv = motor_signal_to_q_qv(q_qv_motor)
    else:
        print(len(read_byte_vec))
        print('The data length is incorrect!')
        q_qv = np.zeros(4)
    return q_qv


def q_qv_to_motor_signal(q_qv_d, k_float_2_int=100, b_float_2_int=30000):
    return (k_float_2_int * q_qv_d + b_float_2_int).astype(np.uint16)  # 100 * (q_k_d, qv_k_d, q_a_d,qv_a_d)


def motor_signal_to_q_qv(q_qv, k_float_2_int=100, b_float_2_int=30000):
    return (q_qv.astype(float) - b_float_2_int) / k_float_2_int  # 100 * (q_k_d, qv_k_d, q_a_d,qv_a_d)



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


def initialize_prosthesis(ser, q_d):
    q_qv_d = np.zeros(4)
    q_qv_d[::2] = q_d[-1]
    step_num = 100
    for i in range(step_num):
        q_qv_d[::2] = q_d[-1] * i / 100
        send_signals_to_motor(ser, q_qv_d)
        time.sleep(8e-3)


def load_gait_paras(mode='level_ground'):
    '''
    Available modes: 'level_ground', 'stair_ascent', 'stair_descent', 'ramp_ascent', 'ramp_descent'
    '''
    gait_paras_dict = np.load('data/paras/gait_paras_dict_{}.npy'.format(mode), allow_pickle=True).item()
    f_joint_angle = gait_paras_dict['f_joint_angle']
    return gait_paras_dict, f_joint_angle


def view_joint_angle(f_joint_angle):
    plt.plot(f_joint_angle(np.linspace(0, 100, 101)) - f_joint_angle(0))
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
