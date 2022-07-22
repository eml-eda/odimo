# DISCLAIMER:
# The integration of different HW models is currently not impemented,
# the proposed MPIC model is only an example but the current implementation
# directly support only `diana`

# TODO: Understand how model changes for deptwhise conv.
#       At this time groups is not taken into account!

MPIC = {
    2: {2: 6.5, 4: 4., 8: 2.2},
    4: {2: 3.9, 4: 3.5, 8: 2.1},
    8: {2: 2.5, 4: 2.3, 8: 2.1},
}

DIANA_NAIVE = {
    'digital': 1.0,
    'analog': 5.0,  # Default SpeedUp
}


def _analog_cycles(**kwargs):
    raise NotImplementedError


def _digital_cycles(**kwargs):
    raise NotImplementedError


def mpic_model(a_bit, w_bit):
    return MPIC[a_bit][w_bit]


def diana_naive(analog_speedup=5.):

    def diana_model(accelerator, **kwargs):
        ch_in = kwargs['ch_in']
        ch_eff = kwargs['ch_out']
        k_x = kwargs['k_x']
        k_y = kwargs['k_y']
        out_x = kwargs['out_x']
        out_y = kwargs['out_y']
        mac = ch_in * ch_eff * k_x * k_y * out_x * out_y
        DIANA_NAIVE['analog'] = float(analog_speedup)  # Update SpeedUp
        if accelerator in DIANA_NAIVE.keys():
            return mac / DIANA_NAIVE[accelerator]
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model


def diana():

    def diana_model(accelerator, **kwargs):
        if accelerator == 'analog':
            return _analog_cycles(**kwargs)
        elif accelerator == 'digital':
            return _digital_cycles(**kwargs)
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model
