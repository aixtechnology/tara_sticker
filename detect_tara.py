import sys
import time
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_requirements, colorstr, non_max_suppression, scale_coords,increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_sync



@torch.no_grad()
def run(weights='yolov5s.pt',
        source='data/images',
        imgsz=640,
        conf_thres=0.25,
        iou_thres=0.45,
        project='tara',
        ):

    detect_area_c1 = (150, 150)
    detect_area_c2 = (600, 400)

    save_dir = increment_path(Path(project) / 'inference', exist_ok=True)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    device = select_device(0)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    names = model.module.names if hasattr(model, 'module') else model.names
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    vid_path, vid_writer = [None] * 1, [None] * 1
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        txt_dis = ""
        txt_dis_color = [0, 255, 0]

        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False, visualize=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=100)

        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = f'{names[c]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=3)

                    if xyxy[1] < detect_area_c1[1]:
                        if c == 1:
                            txt_dis = "NG"
                            txt_dis_color = [0, 0, 255]
                        else:
                            txt_dis = "OK"
                            txt_dis_color = [0, 255, 0]

            cv2.putText(im0, txt_dis, (130, 120), 0, 4, txt_dis_color, thickness=5, lineType=cv2.LINE_AA)
            cv2.rectangle(im0, detect_area_c1, detect_area_c2, color=(128, 128, 128), thickness=3, lineType=cv2.LINE_AA)

            t2 = time_sync()
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

run(weights='runs/train/exp11/weights/last.pt', source='videos/test.mp4')
