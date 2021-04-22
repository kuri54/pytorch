import io
import numpy as np
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Tensorboard ログファイル
path = './tensorboard_runs/image_test_runs/metrics_test_model_multiple_ft/events.out.tfevents.1619053425.29d7a9e53cdb.680.2' 

event_acc = EventAccumulator(path, size_guidance={'images': 0})
event_acc.Reload()
event_acc.Tags()['images']

for tag in event_acc.Tags()['images']:
    events = event_acc.Images(tag)
    tag_name = tag.replace('/', '_')
    for index, event in enumerate(events):
        # 画像はエンコードされているので戻す
        s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)

        img_from_str = Image.open(io.BytesIO(event.encoded_image_string))
        img_from_str.save('test_image_{}.png'.format(tag_name))
