MPIC = {
    2: {2: 6.5, 4: 4., 8: 2.2},
    4: {2: 3.9, 4: 3.5, 8: 2.1},
    8: {2: 2.5, 4: 2.3, 8: 2.1},
}

DIANA = {
    'digital': 1.0,
    'analog': 5.0,
}

def mpic_model(a_bit, w_bit):
    return MPIC[a_bit][w_bit]

def diana(analog_speedup=5.):
    def diana_model(accelerator):
        DIANA['analog'] = float(analog_speedup)
        if accelerator in DIANA.keys():
            return DIANA[accelerator]
        else:
            raise ValueError(f'Unknown accelerator: {accelerator}')

    return diana_model