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
    out1 = dec(outputs["ids"], outputs, False)
    out2 = pos(outputs['poses'])
    break
```
