import argparse
import torch

import models

parser = argparse.ArgumentParser(description='Get model size')
parser.add_argument('arch', type=str, help='Architecture name')
parser.add_argument('--num-classes', type=int, help='Number of output classes')
regularizer_targets = ['latency', 'power']
parser.add_argument('--target', '--t', type=str,
                    choices=regularizer_targets,
                    help=f'regularization target: {*regularizer_targets,}')
parser.add_argument('--input-res', type=int, default=None,
                    help='Input Resolution (used only for res18)')
parser.add_argument('--no-std-head', default=True, action='store_false', dest='std_head',
                    help='Whether to use std-head (used only for res18)')
parser.add_argument('--analog-speedup', type=float, default=5.,
                    help='SpeedUp of analog wrt digital')
parser.add_argument('--pretrained-model', type=str, default=None, help='Pretrained model path')

args = parser.parse_args()
print(args)

model_name = str(args.arch).split('quant')[1].split('_')[0]

# Build and Load pretrained model if specified
if args.pretrained_model is not None:
    if 'mix' in args.arch:
        model = models.__dict__[args.arch](
            args.pretrained_model, num_classes=args.num_classes,
            )
    else:
        if model_name == 'res18':
            model = models.__dict__[args.arch](
                args.pretrained_model, num_classes=args.num_classes,
                fine_tune=False, analog_speedup=args.analog_speedup,
                std_head=args.std_head, target=args.target)
        else:
            model = models.__dict__[args.arch](
                args.pretrained_model, analog_speedup=args.analog_speedup,
                num_classes=args.num_classes, fine_tune=False, target=args.target)
else:
    model = models.__dict__[args.arch](
        '', num_classes=args.num_classes, target=args.target)

# Feed random input
if model_name == 'mobilenetv1':
    rnd_input = torch.randn(1, 3, 96, 96)
elif model_name in ['res20', 'res8']:
    rnd_input = torch.randn(1, 3, 32, 32)
elif model_name == 'res18':
    if args.input_res is None or args.input_res == 64:
        rnd_input = torch.randn(1, 3, 64, 64)
    else:
        rnd_input = torch.randn(1, 3, 224, 224)
elif model_name == 'dscnn':
    rnd_input = torch.randn(1, 1, 49, 10)
elif model_name == 'denseae':
    rnd_input = torch.randn(2, 640, 1, 1)
elif model_name == 'temponet':
    rnd_input = torch.randn(2, 4, 256)
else:
    raise ValueError(f'Unknown model name: {model_name}')
with torch.no_grad():
    model(rnd_input)

cycles, bita, bitw = model.fetch_arch_info()

print(f'{args.target}: {cycles}')
print(f'bita: {bita}')
print(f'bitw: {bitw}')
