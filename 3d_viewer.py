import numpy as np
import time
import cv2
import glob
import shutil
import open3d as o3d
import matplotlib.pyplot as plt
from Utils import Plot, Algo, IO

def test_offline_imu_reader(show_camera = False):
    fig, line_list = Plot.phase_estimation_init_plot()
    for i in range(1000000):
        try:
            '''Show IMU data'''
            data_mat = np.load('data/estimated_walking.npy')
            phase_vec = data_mat[-101:, -4]
            joint_angle_vec = data_mat[-101:, -3:]
            phase_vec_list = [phase_vec, phase_vec]
            joint_angle_vec_list = [joint_angle_vec, joint_angle_vec]
            Plot.phase_estimation_update_plot(fig, line_list, phase_vec_list, joint_angle_vec_list, i, is_savefig= False)
            if show_camera:
                '''Show camera data'''
                img_names = glob.glob('data/local_camera/*.jpg')
                img_name = sorted(img_names)[-1]
                img = cv2.imread(img_name, -1)
                cv2.imshow("Local camera", img)
                cv2.waitKey(10)
        except:
            print('Not finish writting.')
            time.sleep(2e-3)


def calc_transform_mat(roll_angle = 180):
    transform_mat = np.identity(4)
    roll_angle = np.deg2rad(roll_angle)
    rot_mat = np.array([[1, 0, 0],
                        [0, np.cos(roll_angle), -np.sin(roll_angle)],
                        [0, np.sin(roll_angle), np.cos(roll_angle)]])
    transform_mat[:3, :3] = rot_mat
    return transform_mat

def thigh_angle_init_plot():
    fig = plt.figure(figsize=(16, 4))
    thigh_angle_vec = np.zeros(1000)
    line, = plt.plot(thigh_angle_vec)
    plt.xlabel('Time step')
    plt.ylabel('Angle')
    plt.ylim([-150, 100])
    fig.tight_layout()
    plt.pause(0.1)
    return fig, line


def thigh_angle_update_plot(fig, line, thigh_angle_vec):
    line.set_ydata(thigh_angle_vec)
    fig.canvas.draw()
    fig.canvas.flush_events()


def rgbd_viewer():
    viewer_setting_file = 'data/paras/3d_viewer_setting.json'
    vis = Plot.init_o3d_vis()
    thigh_angle_vec = np.zeros(1000)
    fig, line = thigh_angle_init_plot()
    for i in range(100000):
        current_time = time.time()
        try:
            imu_time_vec = np.load('data/IMUOutEvening/time_vec.npy')[-1]
            imu_data_vec = np.load('data/IMUOutEvening/{:.3f}.npy'.format(imu_time_vec))
            thigh_angle_vec = Algo.fifo_data_vec(thigh_angle_vec, imu_data_vec[9])
            rgbd_time_vec = np.load('data/RGBDOutEvening/time_vec.npy')
            # cam_time_vec = np.load('data/web_cam/time_vec.npy')
            img_name = 'data/RGBDOutEvening/{:.3f}'.format(rgbd_time_vec[-1])
            depth_name = '{}.png'.format(img_name)
            rgb_name = '{}.jpg'.format(img_name)
            current_pcd = IO.read_rgbd_pcd(rgb_name, depth_name)
            # current_pcd = IO.read_depth_pcd(depth_name)
            thigh_angle_update_plot(fig, line, thigh_angle_vec)
            vis = Plot.view_o3d_vis(vis, [current_pcd],viewer_setting_file=viewer_setting_file)
            # web_img = cv2.imread('data/web_cam/{:.3f}.jpg'.format(cam_time_vec[-1]))
            # cv2.imshow('rgb', web_img)
            cv2.imshow('rgb', cv2.rotate(cv2.imread(rgb_name), rotateCode=cv2.ROTATE_180))
            k = cv2.waitKey(10)
            if k == 27:  # Esc key to stop
                break
            # if 1 == i:
            #     vis.run()
            #     IO.save_view_point(vis, viewer_setting_file=viewer_setting_file)
            print('Costed time: {:.3f} s'.format(time.time() - current_time))
        except:
            time.sleep(3e-2)


if __name__ == '__main__':
    # test_offline_imu_reader()
    rgbd_viewer()