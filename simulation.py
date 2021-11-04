import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from matplotlib import cm
from Utils import Algo, IO, Plot

matplotlib.use('Qt5Agg')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 12})

def main():
    compare_different_phase_variables()
    # Plot.visualize_hyper_paras_of_sigmoid()
    # Plot.generate_video_of_sigmoid()



def compare_different_phase_variables(is_video = True):
    phase_1 = np.arange(start=0, stop=100)
    phase_2 = 40 * np.ones(75)
    phase_vec = np.r_[phase_1, phase_1[:40], phase_2, phase_1[40:],
                      phase_1[:40], phase_1[40:10:-1], phase_1[10:]]
    thigh_angle_vec = Algo.simulate_thigh_angle(phase_vec)
    knee_angle_vec = simulate_knee_angle(phase_vec)
    predicted_phase_mat = np.zeros((phase_vec.shape[0], 3))
    predicted_knee_angle_mat = np.zeros((knee_angle_vec.shape[0], 3))
    x_mat = np.zeros((phase_vec.shape[0], 3))
    y_mat = np.zeros((phase_vec.shape[0], 3))
    '''1: estimate phase based on q_int_q'''
    predicted_phase_mat[:, 0], x_mat[:, 0], y_mat[:, 0] = Algo.calc_phase_based_on_q_int_q(thigh_angle_vec)
    '''2: estimate phase based on q_dq'''
    predicted_phase_mat[:, 1], x_mat[:, 1], y_mat[:, 1] = Algo.calc_phase_based_on_q_dq(thigh_angle_vec)
    '''3: estimate phase based on q'''
    predicted_phase_mat[:, 2], x_mat[:, 2], y_mat[:, 2] = Algo.calc_phase_based_on_q(thigh_angle_vec)
    for i in range(3):
        predicted_knee_angle_mat[:, i] = simulate_knee_angle(predicted_phase_mat[:, i])

    if is_video:
        plot_all(thigh_angle_vec, knee_angle_vec, phase_vec,
                 predicted_phase_mat, predicted_knee_angle_mat,
                 x_mat, y_mat)
    else:
        plot_figure(thigh_angle_vec, knee_angle_vec, phase_vec,
                 predicted_phase_mat, predicted_knee_angle_mat,
                 x_mat, y_mat)



def plot_figure(thigh_angle_vec, knee_angle_vec, phase_vec, predicted_phase_mat, predicted_knee_angle_mat,
             x_mat, y_mat):
    # fig = plt.figure(figsize=(8, 8))
    cm_fun = cm.get_cmap('tab10', 10)
    method_list = ['q_int_q', 'q_dq', 'q']
    legend_list = ['Actual', r'Predicted by $\phi-\int \phi$', r'Predicted by $\phi-\dot{\phi}$', 'Predicted by $\phi$']
    line_mat = np.zeros((5, 3), dtype=np.object)
    y_label_list = [r'Thigh angle integration ($\degree$ s)',
                    r'Thigh angle velocity ($\degree$/s)',
                    r'Thigh angle ($\degree$)', ]
    time_vec = np.arange(len(thigh_angle_vec)) / 100 # unit: s
    ''' 2. Predicted phase '''
    # plt.subplot(3, 1, 2)
    # plt.plot(phase_vec, color=cm_fun(0), linewidth=2) # Actual phase
    # for c in range(3):
    #     plt.plot(predicted_phase_mat[:, c], '--', color=cm_fun(c+1), linewidth=2) # predicted phase
    # plt.ylabel('Gait cycle (%)')
    # '''
    #     1. Thigh angle curve
    # '''
    # plt.subplot(3, 1, 1)
    # plt.plot(time_vec, thigh_angle_vec, linewidth=2)
    # plt.ylabel(r'Thigh angle $(\degree)$')
    # ''' 3. Predicted knee angle '''
    # plt.subplot(3, 1, 3)
    # plt.plot(knee_angle_vec, color=cm_fun(0), linewidth=2)  # Actual knee angle
    # for c in range(3):
    #     plt.plot(predicted_knee_angle_mat[:, c], '--', color=cm_fun(c + 1), linewidth=2)  # predicted phase
    # plt.ylabel('Gait cycle (%)')
    # plt.xlabel('Time (s)')
    # fig.legend(legend_list, loc='lower center',
    #            ncol=4, bbox_to_anchor=(0.50, 0.97), frameon=False)
    # fig.tight_layout()
    # plt.savefig('results/images/simulation.pdf', bbox_inches='tight')

    fig = plt.figure(figsize=(8, 4))
    cm_fun = cm.get_cmap('cool', len(x_mat[:, 0]))
    for c in range(2):
        plt.subplot(1, 2, c+1)
        for i in range(len(x_mat[:, c])-1):
            plt.plot(x_mat[i:i+2, c], y_mat[i:i+2, c], color=cm_fun(i), linewidth=2)
        plt.xlabel(y_label_list[-1])
        plt.ylabel(y_label_list[c])
    fig.tight_layout()
    plt.savefig('results/images/phaseCircle.pdf', bbox_inches='tight')

    # for c in range(3):
    #     plt.subplot(3, 5, c * 5 + 1)
    #     line_mat[0, c], = plt.plot(thigh_angle_vec, color=cm_fun(1))
    #     plt.ylabel(r'Thigh angle $(\degree)$')
    #     plt.xlabel('Time step')
    #
    #     plt.subplot(3, 5, c * 5 + 2)
    #     line_mat[1, c], = plt.plot(predicted_phase_mat[:, c], color=cm_fun(1), linewidth=3)
    #     plt.plot(phase_vec, '--', color=cm_fun(0), linewidth=1)
    #     plt.ylabel(r'Gait cycle (%)')
    #     plt.ylim([-1, 101])
    #     plt.xlabel('Time step')
    #
    #     plt.subplot(3, 5, c * 5 + 3)
    #     line_mat[2, c], = plt.plot(predicted_knee_angle_mat[:, c], color=cm_fun(1), linewidth=3)
    #     plt.plot(knee_angle_vec, '--', color=cm_fun(0), linewidth=1)
    #     plt.ylabel(r'Knee angle $(\degree)$')
    #     plt.xlabel('Time step')
    #     plt.ylim([-1, 101])
    #     plt.title('Method: {}'.format(method_list[c]))
    #
    #     plt.subplot(3, 5, c * 5 + 4)
    #     plt.plot(x_mat[:, c], y_mat[:, c], '.', color=cm_fun(0), markersize=2)
    #     line_mat[3, c], = plt.plot(x_mat[:, c], y_mat[:, c], 'o', color=cm_fun(1), markersize=5)
    #     if c < 2:
    #         plt.xlabel(r'Thigh angle $(\degree)$')
    #     else:
    #         plt.xlabel(r'Gait cycle (%)')
    #
    #     plt.ylabel(y_label_list[c])
    #     plt.subplot(3, 5, c * 5 + 5)
    #     line_mat[4, c] = plot_leg(thigh_angle_vec[-1], knee_angle_vec[-1], predicted_knee_angle_mat[-1, c], cm_fun, )

    plt.show()


def plot_all(thigh_angle_vec, knee_angle_vec, phase_vec, predicted_phase_mat, predicted_knee_angle_mat,
             x_mat, y_mat):
    fig = plt.figure(figsize=(16, 9))
    cm_fun = cm.get_cmap('tab10', 10)
    method_list = ['q_int_q', 'q_dq', 'q']
    line_mat = np.zeros((5, 3), dtype=np.object)
    y_label_list = [r'Thigh angle integration ($\degree$ s)',
                    r'Thigh angle velocity ($\degree$/s)',
                    r'Thigh angle ($\degree$)', ]
    for c in range(3):
        plt.subplot(3, 5, c * 5 + 1)
        line_mat[0, c], = plt.plot(thigh_angle_vec, color=cm_fun(1))
        plt.ylabel(r'Thigh angle $(\degree)$')
        plt.xlabel('Time step')

        plt.subplot(3, 5, c * 5 + 2)
        line_mat[1, c], = plt.plot(predicted_phase_mat[:, c], color=cm_fun(1), linewidth=3)
        plt.plot(phase_vec, '--', color=cm_fun(0), linewidth=1)
        plt.ylabel(r'Gait cycle (%)')
        plt.ylim([-1, 101])
        plt.xlabel('Time step')

        plt.subplot(3, 5, c * 5 + 3)
        line_mat[2, c], = plt.plot(predicted_knee_angle_mat[:, c], color=cm_fun(1), linewidth=3)
        plt.plot(knee_angle_vec, '--', color=cm_fun(0), linewidth=1)
        plt.ylabel(r'Knee angle $(\degree)$')
        plt.xlabel('Time step')
        plt.ylim([-1, 101])
        plt.title('Method: {}'.format(method_list[c]))

        plt.subplot(3, 5, c * 5 + 4)
        plt.plot(x_mat[:, c], y_mat[:, c], '.', color=cm_fun(0), markersize=2)
        line_mat[3, c], = plt.plot(x_mat[:, c], y_mat[:, c], 'o', color=cm_fun(1), markersize=5)
        if c < 2:
            plt.xlabel(r'Thigh angle $(\degree)$')
        else:
            plt.xlabel(r'Gait cycle (%)')

        plt.ylabel(y_label_list[c])
        plt.subplot(3, 5, c * 5 + 5)
        line_mat[4, c] = plot_leg(thigh_angle_vec[-1], knee_angle_vec[-1], predicted_knee_angle_mat[-1, c], cm_fun,)
    fig.tight_layout()
    plt.show()
    # img_dir = 'results/simulation_imgs'
    # for i in np.arange(0, len(thigh_angle_vec), step=3):
    #     print(i)
    #     for c in range(3):
    #         y_data_list = [thigh_angle_vec[:i], predicted_phase_mat[:i, c], predicted_knee_angle_mat[:i, c]]
    #         for r in range(3):
    #             line_mat[r, c].set_xdata(np.arange(i))
    #             line_mat[r, c].set_ydata(y_data_list[r])
    #         line_mat[3, c].set_xdata(x_mat[i, c])
    #         line_mat[3, c].set_ydata(y_mat[i, c])
    #         plot_leg(thigh_angle_vec[i], knee_angle_vec[i], predicted_knee_angle_mat[i, c], cm_fun, is_plot=False,
    #                  lines=line_mat[4, c])
    #     fig.canvas.draw()
    #     fig.canvas.flush_events()
    #     plt.savefig('{}/{:03d}.png'.format(img_dir, i), bbox_inches='tight')


def plot_leg(thigh_angle, knee_angle, predicted_knee_angle, cm_fun, is_plot=True, lines=None):
    thigh_angle *= np.pi / 180
    knee_angle *= np.pi / 180
    predicted_knee_angle *= np.pi / 180
    x_vec = np.array([0, np.sin(thigh_angle), np.sin(thigh_angle) + np.sin(thigh_angle - knee_angle)])
    y_vec = np.array([0, -np.cos(thigh_angle), -(np.cos(thigh_angle) + np.cos(thigh_angle - knee_angle))])
    predicted_x_vec = np.copy(x_vec)
    predicted_x_vec[-1] = np.sin(thigh_angle) + np.sin(thigh_angle - predicted_knee_angle)
    predicted_y_vec = np.copy(y_vec)
    predicted_y_vec[-1] = -(np.cos(thigh_angle) + np.cos(thigh_angle - predicted_knee_angle))
    if is_plot:
        lines = np.zeros(2, dtype=np.object)
        lines[1], = plt.plot(predicted_x_vec, predicted_y_vec, color=cm_fun(1), linewidth=3)
        lines[0], = plt.plot(x_vec, y_vec, '--', color=cm_fun(0), linewidth=3)
        plt.axis('square')
        plt.ylim([-2, 2])
        plt.xlim([-2, 2])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.legend(['Predicted', 'Actual'], frameon=False, handlelength=2)
    else:
        print(lines)
        lines[0].set_xdata(x_vec)
        lines[0].set_ydata(y_vec)
        lines[1].set_xdata(predicted_x_vec)
        lines[1].set_ydata(predicted_y_vec)
    return lines


def simulate_knee_angle(phase):
    phi = 2 * np.pi * 1e-2 * phase  # [0, np.pi]
    knee_angle = 50 * np.cos(phi + np.pi / 3) + 50
    return knee_angle

def write_video():
    img_name_vec = glob.glob('results/simulation_imgs/*.png')
    video_name = 'results/simulation.mp4'
    IO.read_image_to_video(img_name_vec, video_name, fps=10)




if __name__ == '__main__':
    a = 1
    #main()
    # write_video()
