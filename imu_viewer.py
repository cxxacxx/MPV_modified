import numpy as np
import time
import cv2
import copy
import matplotlib.pyplot as plt
from Utils import Algo


def signal_init_plot():
    '''
    Thigh angle: pitch.
    Thigh acceleration: acc_y.
    '''
    fig = plt.figure(figsize=(8, 4))
    signals = np.zeros((200, 6))
    line_list = []
    idx_list = [1, 4]
    y_label_list = ['Roll (deg)', 'Pitch (deg)', 'Yaw (deg)', 'Acc_x (m/s^2)','Acc_y (m/s^2)','Acc_z (m/s^2)']
    for i in range(2):
        plt.subplot(1, 2, i+1)
        line, = plt.plot(signals[:, idx_list[i]])
        plt.xlabel('Time step')
        plt.ylabel(y_label_list[idx_list[i]])
        plt.ylim([-50, 50])
        line_list.append(line)
    fig.tight_layout()
    plt.pause(0.1)
    return fig, line_list


def signal_update_plot(fig, line_list, signals):
    idx_list = [1, 4]
    for i in range(2):
        line_list[i].set_ydata(signals[:, idx_list[i]])
    fig.canvas.draw()
    fig.canvas.flush_events()


def imu_viewer():
    imu_data_mat = np.zeros((200, 6))
    data_vec = np.memmap('log/imu_euler_acc.npy', dtype='float32', mode='r', shape=(6,))
    fig, line_list = signal_init_plot()
    for i in range(100000):
        current_time = time.time()
        # try:
        imu_data_vec = np.zeros(6)
        imu_data_vec[:] = data_vec[:]
        if len(imu_data_vec == 6):
            imu_data_mat = Algo.fifo_data_vec(imu_data_mat, imu_data_vec)
        if i % 10 == 0:
            signal_update_plot(fig, line_list, imu_data_mat)
        print('Costed time: {:.3f} s'.format(time.time() - current_time))
        # except:
        #     time.sleep(1e-3)
        time.sleep(1e-2)


if __name__ == '__main__':
    imu_viewer()