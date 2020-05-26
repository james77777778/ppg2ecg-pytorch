import os
import re
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from biosppy import ecg
from biosppy.utils import normpath


MAJOR_LW = 2.5
MINOR_LW = 1.5
RPEAK_MAXLOSS = 75  # ms


def dict2table(params):
    if not isinstance(params, dict):
        params = vars(params)
    text = '\n\n'
    text = '|  Attribute  |     Value    |\n'+'|'+'-'*13+'|'+'-'*14+'|'
    for key, value in params.items():
        text += '\n|{:13}|{:14}|'.format(str(key), str(value))
    return text


def signal2waveform(signal, x_size=300, y_size=240):
    # assume signal value range: [-1, 1]
    img_y_size = y_size
    img_x_size = x_size
    img = np.zeros((img_y_size, img_x_size))
    sig_x_step = len(signal)/x_size

    for index in range(img_x_size):
        amp = signal[int(index*sig_x_step)]
        value = img_y_size - int((amp+1)*y_size//2)
        value = np.clip(value, 0, y_size-1)
        pre_index = max(0, index-1)
        pre_amp = signal[int(pre_index*sig_x_step)]
        pre_value = img_y_size - int((pre_amp+1)*y_size//2)
        pre_value = np.clip(pre_value, 0, y_size-1)
        value_range = int(np.abs(pre_value-value))
        try:
            for j in range(value_range):
                if value > pre_value:
                    if j < value_range//2:
                        img[pre_value+j][pre_index] = 1
                    else:
                        img[pre_value+j][index] = 1
                else:
                    if j < value_range//2:
                        img[pre_value-j][pre_index] = 1
                    else:
                        img[pre_value-j][index] = 1
            img[value][index] = 1
        except IndexError as e:
            print(e)
            print(img.shape)
            print(pre_value, j)
            print(pre_index, index)
    return img


def get_bidmc_num(basename):
    regex = re.compile(r'\d+')
    nums = [int(x) for x in regex.findall(basename)]
    return nums[0]


def make_filter_bidmc_num(exclude_id):
    def filter_bidmc_num(basename):
        regex = re.compile(r'\d+')
        nums = [int(x) for x in regex.findall(basename)]
        if nums[0] in exclude_id:
            return False
        else:
            return True
    return filter_bidmc_num


def make_filter_tbme_num(case_id):
    def filter_tbme_num(basename):
        regex = re.compile(r'\d+')
        nums = [int(x) for x in regex.findall(basename)]
        if nums[2] == case_id:
            return True
        else:
            return False
    return filter_tbme_num


def plot_ecg(ts, ori_ecg, gen_ecg, ori_rpeaks, gen_rpeaks, templates_ts,
             ori_templates, gen_templates, path, ppg=None, attn_out=None):
    fig = plt.figure()
    # fig.suptitle('ECG Summary')
    gs = gridspec.GridSpec(4, 2)
    # original signal
    ax1 = fig.add_subplot(gs[:2, 0])
    ymin = np.min(ori_ecg)
    ymax = np.max(ori_ecg)
    alpha = 0.1 * (ymax - ymin)
    ymax += alpha
    ymin -= alpha
    ax1.plot(ts, ori_ecg, linewidth=MAJOR_LW, label='Groundtruth')
    ax1.vlines(ts[ori_rpeaks], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='R-peaks')
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Magnitude (mV)')
    ax1.set_title('Groundtruth ECG')
    ax1.grid()
    # generated
    ax2 = fig.add_subplot(gs[2:, 0], sharex=ax1)
    ax2.plot(ts, gen_ecg, linewidth=MAJOR_LW, label='Reconstructed')
    ax2.vlines(ts[gen_rpeaks], ymin, ymax,
               color='m',
               linewidth=MINOR_LW,
               label='R-peaks')
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Magnitude (mV)')
    ax2.set_title('Reconstructed ECG')
    ax2.grid()
    # templates
    ax4 = fig.add_subplot(gs[2:5, 1])
    ax4.plot(
        templates_ts, ori_templates.T, 'm', linewidth=MINOR_LW, alpha=0.7,
        label='Groundtruth')
    try:
        ax4.plot(
            templates_ts, gen_templates.T, 'c', linewidth=MINOR_LW, alpha=0.7,
            label='Reconstructed')
    except ValueError:
        print(templates_ts)
        print(gen_templates.T)
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Magnitude (mV)')
    ax4.set_title('Templates')
    handles, labels = ax4.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax4.legend(by_label.values(), by_label.keys())
    ax4.grid()
    # ppg and attn_out
    if ppg is not None:
        ax5 = fig.add_subplot(gs[0:2, 1])
        ax5.plot(ts, ppg, linewidth=MAJOR_LW, label='PPG')
        if((attn_out is not None)):
            ax5.plot(
                ts, attn_out, color='m', linewidth=MINOR_LW,
                label='attn_out')
        ax5.set_title('Reference PPG')
        ax5.set_xlabel('Time (sec)')
        ax5.set_ylabel('Magnitude (mV)')
        ax5.grid()

    # make layout tight
    gs.tight_layout(fig)
    # save to file
    if path is not None:
        path = normpath(path)
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext not in ['png', 'jpg']:
            path = root + '.png'
        fig.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def rpeak_plot(ori_ecg, ori_rpeaks, gen_ecg, gen_rpeaks, sampling_rate=100.,
               fig_path=None, ppg=None, attn_out=None):
    # templates for ori_ecg
    ori_templates, rpeaks_ = ecg.extract_heartbeats(
        signal=ori_ecg,
        rpeaks=ori_rpeaks,
        sampling_rate=sampling_rate,
        before=0.2,
        after=0.4)
    # templates for gen_ecg
    gen_templates, rpeaks_ = ecg.extract_heartbeats(
        signal=gen_ecg,
        rpeaks=gen_rpeaks,
        sampling_rate=sampling_rate,
        before=0.2,
        after=0.4)
    # plot
    # truncate size
    # SAMPLES = 200
    # LEFT_TRUC = int(SAMPLES*0.1)
    # RIGHT_TRUC = int(SAMPLES*0.1)
    length = len(ori_ecg)
    T = (length - 1) / sampling_rate
    ts = np.linspace(0, T, length, endpoint=True)
    try:
        ts_tmpl = np.linspace(
            -0.2, 0.4, ori_templates.shape[1], endpoint=False)
        plot_ecg(
            ts, ori_ecg, gen_ecg, ori_rpeaks, gen_rpeaks, ts_tmpl,
            ori_templates, gen_templates, fig_path, ppg=ppg, attn_out=attn_out)
    except IndexError:
        print("\nno heartbeat found at: {}".format(os.path.basename(fig_path)))


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def rpeak_metric(ori_rpeaks, gen_rpeaks, ori_ecg, gen_ecg, sample_rate=100.):
    total_pos_loss = 0
    total_mag_loss = 0
    failed = 0
    num_rpeaks = 0
    pos_loss_arr = []
    mag_loss_arr = []
    for ori_value in ori_rpeaks:
        if len(gen_rpeaks) > 0:
            nearest_gen_value = find_nearest(gen_rpeaks, ori_value)
            # calculate r-peak position loss
            l1_loss = np.abs(ori_value - nearest_gen_value)
        else:
            l1_loss = (RPEAK_MAXLOSS+1)*sample_rate/1000.  # always go failed
        # r-peak detection failed
        if l1_loss*1000./sample_rate > RPEAK_MAXLOSS:  # 5*0.01 = 50ms
            failed += 1
            num_rpeaks += 1
            total_pos_loss += RPEAK_MAXLOSS*sample_rate/1000.
            pos_loss_arr.append(RPEAK_MAXLOSS*sample_rate/1000.)
        # r-peak detection success
        else:
            num_rpeaks += 1
            total_pos_loss += l1_loss
            pos_loss_arr.append(ori_value - nearest_gen_value)
            # calculate r-peak magnitude loss
            mag_loss = np.abs(ori_ecg[ori_value] - gen_ecg[nearest_gen_value])
            total_mag_loss += mag_loss
            mag_loss_arr.append(
                ori_ecg[ori_value] - gen_ecg[nearest_gen_value])
    result = {}
    result['r_pos_error'] = total_pos_loss
    result['r_mag_error'] = total_mag_loss
    result['r_pos_error_arr'] = pos_loss_arr
    result['r_mag_error_arr'] = mag_loss_arr
    result['num_rpeaks'] = num_rpeaks
    result['failed'] = failed
    return result


def rpeak_detection(ecg_data, sampling_rate=100.):
    # rpeak segmentation for ori_ecg
    rpeaks, = ecg.hamilton_segmenter(
        signal=ecg_data, sampling_rate=sampling_rate)
    rpeaks, = ecg.correct_rpeaks(
        signal=ecg_data,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        tol=0.05)
    return rpeaks
