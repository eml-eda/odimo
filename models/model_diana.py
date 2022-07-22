import numpy as np
import matplotlib.pyplot as plt

F = 260000000  # Hz


def _floor(ch, N):
    return np.floor((ch + N - 1) / N)


def _ox_unroll_base(ch_in, ch_out, fs_x, fs_y):
    ox_unroll_base = 1
    channel_input_unroll = 64 if ch_in < 64 else ch_in
    for ox_unroll in [1, 2, 4, 8]:
        if (ch_out * ox_unroll <= 512) and \
                (channel_input_unroll * fs_y * (fs_x + ox_unroll - 1) <= 1152):
            ox_unroll_base = ox_unroll
    return ox_unroll_base


def digital_cycles(ch_in, ch_out, out_x, out_y, fs_x, fs_y):
    cycles = _floor(ch_out, 16) * ch_in * _floor(out_x, 16) * out_y * fs_x * fs_y
    cycles_load_store = out_x * out_y * (ch_out + ch_in) / 8
    MACs = ch_in * ch_out * out_x * out_y * fs_x * fs_y
    MAC_cycles = MACs / (cycles + cycles_load_store)
    return MAC_cycles, (cycles + cycles_load_store)


def analog_cycles(ch_in, ch_out, out_x, out_y, fs_x, fs_y):
    # ox_unroll_base = 1
    # channel_input_unroll = 64 if ch_in < 64 else ch_in
    # for ox_unroll in [1, 2, 4, 8]:
    #     if (ch_out * ox_unroll <= 512) and \
    #             (channel_input_unroll * fs_y * (fs_x + ox_unroll - 1) <= 1152):
    #         ox_unroll_base = ox_unroll
    ox_unroll_base = _ox_unroll_base(ch_in, ch_out, fs_x, fs_y)
    cycles_computation = _floor(ch_out, 512) * _floor(ch_in, 128) * out_x * out_y / ox_unroll_base
    # if channel_output <= 256:
    #   cycles_weights = 3 * 2 * fs_x * fs_y * channel_input
    # else:
    # 	cycles_weights = 4 * 2 * fs_x * fs_y * channel_input
    cycles_weights = 4 * 2 * 1152
    MACs = ch_in * ch_out * out_x * out_y * fs_x * fs_y
    MAC_cycles = MACs / ((cycles_computation * 70 / (1000000000 / F) + cycles_weights))
    return MAC_cycles, (cycles_computation * 70 / (1000000000 / F) + cycles_weights)


if __name__ == '__main__':
    analog = []
    digital = []
    analog_cyc = []
    digital_cyc = []
    ox_unroll = []
    ch_max = 256
    for ch in np.arange(1, ch_max):
        MAC_cycles_digital, cycles_digital = digital_cycles(32, ch, 16, 16, 3, 3)
        MAC_cycles_analog, cycles_analog = analog_cycles(32, ch, 16, 16, 3, 3)
        analog.append(MAC_cycles_analog)
        digital_cyc.append(cycles_digital)
        analog_cyc.append(cycles_analog)
        digital.append(MAC_cycles_digital)
        ox_unroll.append(_ox_unroll_base(32, ch, 3, 3))
    plt.plot(np.arange(1, ch_max), analog, label="analog")
    plt.plot(np.arange(1, ch_max), digital, label="digital")
    plt.legend()
    plt.savefig("MAC_cycles.png")
    plt.figure()
    plt.plot(np.arange(1, ch_max), ox_unroll)
    plt.savefig("ox_unroll.png")

    plt.figure()
    plt.plot(np.arange(1, ch_max), analog_cyc, label="analog")
    plt.plot(np.arange(1, ch_max), digital_cyc, label="digital")
    plt.legend()
    plt.savefig("cycles.png")
