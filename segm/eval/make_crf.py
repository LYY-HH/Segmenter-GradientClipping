import joblib
import os
import numpy as np
import argparse
import torch
from PIL import Image
import pandas as pd
from segm.eval.densecrf import crf_inference
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--list",
                        default='data/voc12/ImageSets/Segmentation/val.txt',
                        type=str)
    parser.add_argument("--predict-dir", default=None, type=str)
    parser.add_argument("--predict-png-dir", default=None, type=str)
    parser.add_argument("--img-path", default=None, type=str)
    parser.add_argument("--n-jobs", default=10, type=int)

    args = parser.parse_args()

    df = pd.read_csv(args.list, names=['filename'], converters={"filename": str})
    name_list = df['filename'].values
    if args.predict_png_dir is not None:
        Path(args.predict_png_dir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        def compute(i):
            path = os.path.join(args.predict_dir, name_list[i] + ".npy")
            seg_prob = np.load(path, allow_pickle=True).item()
            keys = seg_prob["keys"]
            seg_prob = torch.tensor(seg_prob["prob"])

            orig_image = np.asarray(Image.open(os.path.join(args.img_path, name_list[i] + ".jpg")).convert("RGB"))
            # seg_pred = torch.nn.functional.softmax(seg_prob, dim=0).cpu().numpy()
            seg_pred = seg_prob.cpu().numpy()
            seg_pred = crf_inference(orig_image, seg_pred, labels=seg_pred.shape[0])
            # seg_pred = torch.tensor(crf_score)

            seg_pred = np.argmax(seg_pred, axis=0)
            seg_pred = keys[seg_pred].astype(np.uint8)

            predict_img = Image.fromarray(seg_pred)
            predict_img.save(os.path.join(args.predict_png_dir, name_list[i] + ".png"))


        joblib.Parallel(n_jobs=args.n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(compute)(i) for i in range(len(name_list))]
        )
