# 291f
- Run torch version:download pkl file to data/tf_records
```
cd experiments/chair_unsupervised
python ../../dpc/run/train_eval_torch.py
```

### How to use the pytorch version Net
```python
import torch
from nets.img_encoder_torch import imgEncoder
from nets.pc_decoder_torch import pcDecoder
from nets.pose_net_torch import poseDecoder

enc = imgEncoder(cfg, channel_number=3, image_size=128)
dec = pcDecoder(cfg, afterConvSize=4096)
pos = poseDecoder(cfg)

for batch_idx, (data, target) in enumerate(load_dataset()):
    print (data.size())
    out0 = enc(data, False)
    out1 = dec(out0["ids"], out0, False)
    out2 = pos(out0['poses'])
    break

print ([(key, out0[key].size()) for key in out0.keys()])
# [('conv_features', torch.Size([2, 4096])), ('z_latent', torch.Size([2, 1024])), ('poses', torch.Size([2, 500])), ('ids', torch.Size([2, 500]))]

print ([(key, out1[key].size()) for key in out1.keys()])
# [('xyz', torch.Size([2, 8000, 3])), ('rgb', torch.Size([2, 8000, 3]))]

print ([(key, out2[key].size()) for key in out2.keys()])
# [('poses', torch.Size([2, 4, 4])), ('pose_student', torch.Size([2, 4])), ('predicted_translation', torch.Size([2, 3]))]
```
