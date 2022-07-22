# DISCLAIMER:
# The integration of different HW models is currently not impemented,
# the proposed MPIC model is only an example but the current implementation
# directly support only `diana`

MPIC = {
    2: {2: 6.5, 4: 4., 8: 2.2},
    4: {2: 3.9, 4: 3.5, 8: 2.1},
    8: {2: 2.5, 4: 2.3, 8: 2.1},
}

DIANA_NAIVE = {
    'digital': 1.0,
    'analog': 5.0,  # Default SpeedUp
}


def mpic_model(a_bit, w_bit):
    return MPIC[a_bit][w_bit]


def diana_naive(analog_speedup=5.):

    def diana_model(ch, accelerator):
        DIANA_NAIVE['analog'] = float(analog_speedup)  # Update SpeedUp
        if accelerator in DIANA_NAIVE.keys():
            return ch / DIANA_NAIVE[accelerator]
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model
