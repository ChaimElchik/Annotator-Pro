import torch
ckpt = torch.load('data/models/checkpoint_best_ema.pth', map_location='cpu', weights_only=False)
args = vars(ckpt.get('args', {})) if hasattr(ckpt.get('args', None), '__dict__') else ckpt.get('args', {})
print(args)
