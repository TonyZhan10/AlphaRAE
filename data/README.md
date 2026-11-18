# Data formats

`rgba_manifest.jsonl` contains one object per RGBA asset:

```json
{"asset_id":"train/example.png","group_id":"example","path":"../assets/example.png","split":"train","alpha_mode":"straight","source":"user_rgba","width":1024,"height":1024}
```

Paths are relative to the manifest. Splits are `train`, `val` or `test`. A `group_id` and an image path may occur in only one split. RGBA PNGs must be 8-bit with an alpha channel and no embedded ICC profile; convert tagged images to untagged sRGB before preparing the manifest. The loader resizes in linear premultiplied RGB plus alpha and pads to the codec's resolution.

`rgb_replay.jsonl` contains RGB training images:

```json
{"id":"background_0001","path":"../backgrounds/0001.jpg","source":"sa1b","split":"train","allow_flip":true}
```

The codec trainer samples RGB replay from the training rows in this manifest. RGB images must be at least the requested resolution (256 for K7 or 1024 for K23). To train without RGB replay, set its probability to zero and record that change to the original protocol.

Datasets and private study records are excluded from Git.
