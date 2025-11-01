"""
train_model.py
--------------
1) Converts a YOLO-style dataset (yolo_dataset/images/{train,val}, yolo_dataset/labels/{train,val})
   into TFRecord files for train and val.
2) Writes a minimal TF OD API pipeline config stub (you must edit model selection/hyperparams).
3) Launches TensorFlow Object Detection API training using model_main_tf2.py.

USAGE (example):
    python backend/train_model.py \
        --yolo_ds ../yolo_dataset \
        --output_dir ../tf_od \
        --model_dir ../tf_od/models/my_experiment \
        --num_steps 10000 \
        --batch_size 4

REQUIREMENTS / NOTES:
- You must install TensorFlow 2.x and the TensorFlow Object Detection API.
  See: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
- You must have the COCO API installed (pycocotools).
- The script expects YOLO dataset layout:
    yolo_dataset/
      images/
        train/*.jpg
        val/*.jpg
      labels/
        train/*.txt   # each line: "class cx cy w h" normalized (YOLO)
        val/*.txt
- It will convert YOLO labels -> Pascal VOC style bboxes and create TFRecords.
- You should edit the generated pipeline config to point to a supported TF OD model checkpoint
  (e.g., EfficientDet or Faster R-CNN backbone) and adjust hyperparameters.
- This script calls `model_main_tf2.py` from the TF OD API. Ensure `PYTHONPATH` includes the
  `models/research` directory (where `model_main_tf2.py` lives).

Security: for big training runs use GPU VM with tf + CUDA installed.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import tensorflow as tf

# ---------------------------
# Helper: read YOLO label and convert to xyxy in pixels
# ---------------------------
def yolo_to_xyxy(yolo_path, img_w, img_h):
    """Read a YOLO .txt file and return list of boxes as [xmin,ymin,xmax,ymax,class_id]."""
    boxes = []
    with open(yolo_path, "r") as f:
        for line in f:
            s = line.strip().split()
            if not s:
                continue
            cls = int(s[0])
            cx, cy, w, h = map(float, s[1:5])
            xmin = (cx - w/2) * img_w
            ymin = (cy - h/2) * img_h
            xmax = (cx + w/2) * img_w
            ymax = (cy + h/2) * img_h
            boxes.append([int(xmin), int(ymin), int(xmax), int(ymax), cls])
    return boxes

# ---------------------------
# Helper: write TFRecord (using tf.train.Example)
# ---------------------------
def create_tf_example(image_path, label_path, class_map):
    img = Image.open(image_path)
    img_w, img_h = img.size
    with open(image_path, "rb") as f:
        encoded = f.read()
    boxes = []
    classes_text = []
    classes = []
    if os.path.exists(label_path):
        for (xmin,ymin,xmax,ymax,cls) in yolo_to_xyxy(label_path, img_w, img_h):
            boxes.append((xmin/img_w, ymin/img_h, xmax/img_w, ymax/img_h))
            classes_text.append(str(class_map.get(cls, "cattle")).encode("utf8"))
            classes.append(int(cls))
    # Build tf.train.Example
    import io
    import base64
    xmins = [b[0] for b in boxes]
    xmaxs = [b[2] for b in boxes]
    ymins = [b[1] for b in boxes]
    ymaxs = [b[3] for b in boxes]

    feature = {
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_h])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_w])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_path.name).encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_path.name).encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example

# ---------------------------
# Convert dataset to TFRecord
# ---------------------------
def convert_yolo_to_tfrecord(yolo_ds: Path, out_dir: Path, class_map):
    images_train = sorted((yolo_ds/"images"/"train").glob("*.*"))
    images_val = sorted((yolo_ds/"images"/"val").glob("*.*"))
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_records(image_list, label_subdir, out_path):
        if not image_list:
            print(f"Warning: no images found for {out_path}")
            return
        with tf.io.TFRecordWriter(str(out_path)) as writer:
            for img_path in image_list:
                label_path = yolo_ds/"labels"/label_subdir/(img_path.stem + ".txt")
                ex = create_tf_example(img_path, label_path, class_map)
                writer.write(ex.SerializeToString())
        print(f"Wrote TFRecord {out_path} ({len(image_list)} images)")

    write_records(images_train, "train", out_dir/"train.record")
    write_records(images_val, "val", out_dir/"val.record")
    # Also write a simple label_map.pbtxt
    label_map = out_dir/"label_map.pbtxt"
    with open(label_map, "w") as f:
        for k, v in class_map.items():
            f.write("item {\n")
            f.write(f"  id: {k+1}\n")   # TF OD uses 1-based ids
            f.write(f"  name: '{v}'\n")
            f.write("}\n")
    print("Wrote label_map.pbtxt ->", label_map)
    return out_dir/"train.record", out_dir/"val.record", label_map

# ---------------------------
# Write a minimal pipeline config stub
# ---------------------------
PIPELINE_TEMPLATE = """
model {
  faster_rcnn {
    num_classes: {num_classes}
    # NOTE: replace the pretrained_checkpoint and other params as needed
  }
}
train_config: {{
  batch_size: {batch_size}
  num_steps: {num_steps}
  fine_tune_checkpoint: "{pretrained_ckpt}"
  fine_tune_checkpoint_type: "detection"
}}
train_input_reader: {{
  tf_record_input_reader {{
    input_path: "{train_record}"
  }}
  label_map_path: "{label_map}"
}}
eval_input_reader: {{
  tf_record_input_reader {{
    input_path: "{val_record}"
  }}
  label_map_path: "{label_map}"
  shuffle: false
  num_readers: 1
}}
"""

def write_pipeline_config(out_dir: Path, train_record, val_record, label_map, num_classes, batch_size, num_steps, pretrained_ckpt):
    cfg = PIPELINE_TEMPLATE.format(
        num_classes=num_classes,
        batch_size=batch_size,
        num_steps=num_steps,
        pretrained_ckpt=pretrained_ckpt,
        train_record=str(train_record),
        val_record=str(val_record),
        label_map=str(label_map)
    )
    cfg_path = out_dir/"pipeline.config"
    cfg_path.write_text(cfg)
    print("Wrote pipeline config:", cfg_path)
    return cfg_path

# ---------------------------
# Main: call TF OD API training
# ---------------------------
def run_training(model_dir: Path, pipeline_config: Path, num_steps):
    # Assumes TF models/research folder is in PYTHONPATH and model_main_tf2.py is available
    env = os.environ.copy()
    cmd = [
        "python",  # uses the python environment where TF OD API is installed
        "-m", "object_detection.model_main_tf2",
        f"--pipeline_config_path={pipeline_config}",
        f"--model_dir={model_dir}",
        f"--num_train_steps={num_steps}",
        "--alsologtostderr"
    ]
    print("Starting training with command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_ds", type=str, required=True, help="Path to YOLO dataset root (images/ labels/)")
    parser.add_argument("--output_dir", type=str, default="tf_od", help="Where to write TFRecords + label_map")
    parser.add_argument("--model_dir", type=str, default="tf_od/models/exp", help="TF OD model_dir for checkpoints/logs")
    parser.add_argument("--num_steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pretrained_ckpt", type=str, default="", help="Path to detection checkpoint for fine-tuning")
    args = parser.parse_args()

    yolo_ds = Path(args.yolo_ds).resolve()
    out_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    # simple one-class map (cattle) - adjust if you have multiple classes
    class_map = {0: "cattle"}

    print("Converting YOLO dataset to TFRecord...")
    train_rec, val_rec, label_map = convert_yolo_to_tfrecord(yolo_ds, out_dir, class_map)

    print("Writing pipeline config stub...")
    pipeline_cfg = write_pipeline_config(out_dir, train_rec, val_rec, label_map, num_classes=len(class_map), batch_size=args.batch_size, num_steps=args.num_steps, pretrained_ckpt=args.pretrained_ckpt or "")

    print("Launching training...")
    run_training(model_dir, pipeline_cfg, args.num_steps)
