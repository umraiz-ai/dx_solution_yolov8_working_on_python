import os
# Force RTSP to use TCP to avoid UDP packet loss/corruption (fixes HEVC/green screen issues)
# This must be set before cv2 parses any video sources using FFmpeg backend
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine
#from dx_engine import Configuration
from packaging import version

import torch
import torchvision
import threading
import time
from ultralytics.utils import ops

class RTSPStreamLoader:
    def __init__(self, path):
        self.path = path
        # Force RTSP to use TCP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)  # minimize internal buffer
        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        self.read_thread = threading.Thread(target=self.update, daemon=True)
        self.count = 0
        
        if not self.cap.isOpened():
             print(f"[Error] Could not open video source {path}")
             self.running = False
        else:
             self.read_thread.start()
             # Wait for first frame
             print("[DEBUG] Waiting for stream to initialize...")
             for i in range(50):
                 if self.frame is not None: 
                     print("[DEBUG] Stream initialized successfully.")
                     break
                 time.sleep(0.1)
                 if i % 10 == 0:
                     print(f"[DEBUG] Waiting for frame... {i}/50")

    def update(self):
        print(f"[DEBUG] RTSPStreamLoader: Reading thread started for {self.path}")
        while self.running:
            # print("[DEBUG] RTSPStreamLoader: Requesting frame...") # verbose
            ret, frame = self.cap.read()
            # print(f"[DEBUG] RTSPStreamLoader: Read done. ret={ret}") # verbose
            if not ret:
                print("[DEBUG] RTSPStreamLoader: failed to read frame, stopping.")
                self.running = False
                break
            with self.lock:
                self.frame = frame
                self.count += 1
            # tiny sleep to prevent CPU hogging if capture is faster than source fps
            # (though cap.read() is usually blocking to source fps)
            
    def read(self):
        with self.lock:
            return self.frame is not None, self.frame

    def release(self):
        self.running = False
        if self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        self.cap.release()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def letter_box(image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=None):
    
    src_shape = image_src.shape[:2] # height, width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])

    ratio = r, r  
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  

    dw /= 2 
    dh /= 2

    if src_shape[::-1] != new_unpad:  
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)  # add border
    if format is not None:
        image_new = cv2.cvtColor(image_new, format)
    
    return image_new, ratio, (dw, dh)    


def all_decode(outputs, layer_config, n_classes):
    ''' slice outputs'''
    decoded_tensor = []
    for i, output in enumerate(outputs):
        output = np.squeeze(output)
        for l in range(len(layer_config[i+1]["anchor_width"])):
            start = l*(n_classes + 5)
            end = start + n_classes + 5
            
            layer = layer_config[i+1]
            stride = layer["stride"]
            grid_size = output.shape[2]
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
            output[start+4:end,...] = sigmoid(output[start+4:end,...])
            cxcy = output[start+0:start+2,...]
            wh = output[start+2:start+4,...]
            cxcy[0,...] = (sigmoid(cxcy[0,...]) * 2 - 0.5 + grid[0]) * stride
            cxcy[1,...] = (sigmoid(cxcy[1,...]) * 2 - 0.5 + grid[1]) * stride
            wh[0,...] = ((sigmoid(wh[0,...]) * 2) ** 2) * layer["anchor_width"][l]
            wh[1,...] = ((sigmoid(wh[1,...]) * 2) ** 2) * layer["anchor_height"][l]
            decoded_tensor.append(output[start+0:end,...].reshape(n_classes + 5, -1))
            
    decoded_output = np.concatenate(decoded_tensor, axis=1)
    decoded_output = decoded_output.transpose(1, 0)
    
    return decoded_output


def transform_box(pt1, pt2, ratio, offset, original_shape):
    dw, dh = offset
    pt1[0] = (pt1[0] - dw) / ratio[0]
    pt1[1] = (pt1[1] - dh) / ratio[1]
    pt2[0] = (pt2[0] - dw) / ratio[0]
    pt2[1] = (pt2[1] - dh) / ratio[1]

    pt1[0] = max(0, min(pt1[0], original_shape[1]))
    pt1[1] = max(0, min(pt1[1], original_shape[0]))
    pt2[0] = max(0, min(pt2[0], original_shape[1]))
    pt2[1] = max(0, min(pt2[1], original_shape[0]))

    return pt1, pt2

def run_example(config):
    # if version.parse(Configuration().get_version()) < version.parse("3.0.0"):
    #     print("DX-RT version 3.0.0 or higher is required. Please update DX-RT to the latest version.")
    #     exit()
    
    #print("[DEBUG] Loading model and config...")
    model_path = config["model"]["path"]
    #print(f"[DEBUG] Model path: {model_path}")
    classes = config["output"]["classes"]
    n_classes = len(classes)
    score_threshold = config["model"]["param"]["score_threshold"]
    layers = config["model"]["param"]["layer"]
    final_output = config["model"]["param"]["final_outputs"][0]
    sources = config["input"]["sources"]

    print("[DEBUG] Creating inference engine...")
    ie = InferenceEngine(model_path)
    print("[DEBUG] Inference engine created.")
    print(f"[DEBUG] Model version: {ie.get_model_version()}")
    if version.parse(ie.get_model_version()) < version.parse('7'):
        print("dxnn files format version 7 or higher is required. Please update/re-export the model.")
        exit()

    print("[DEBUG] Getting output tensor names...")
    tensor_names = ie.get_output_tensor_names()
    #print(f"[DEBUG] Output tensor names: {tensor_names}")
    layer_idx = []
    for i in range(len(layers)):
        for j in range(len(tensor_names)):
            if layers[i]["name"] == tensor_names[j]:
                layer_idx.append(j)
                break
    #print(f"[DEBUG] Layer indices: {layer_idx}")
    if len(layer_idx) == 0:
        raise ValueError(f"[Error] Layer {layers} is not supported !!") 

    input_size = np.sqrt(ie.get_input_size() / 3)
    #print(f"[DEBUG] Input size: {input_size}")
    count = 1
    output_dir = config.get('output_dir', '.')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for source in sources:
        frame_iterator = []
        if source["type"] == "image":
            frame_iterator = [(cv2.imread(source["path"]), source["path"])]
        elif source["type"] in ["rtsp", "video"]:
            def vid_gen(path):
                loader = RTSPStreamLoader(path)
                if not loader.running:
                    return
                
                idx = 0
                while loader.running:
                    ret, frame = loader.read()
                    if ret:
                        yield frame, f"frame_{idx}"
                        idx += 1
                        # Throttle inference loop to avoid busy loop if inference is faster than camera
                        # (unlikely on OrangePi, but good practice)
                    else:
                        time.sleep(0.01)
                loader.release()
            frame_iterator = vid_gen(source["path"])
        else:
            print(f"[Warning] Skipping unknown source type: {source.get('type')}")
            continue

        
        video_writer = None
        start_time = None
        FRAME_DURATION = 20  # seconds

        for image_src, input_desc in frame_iterator:
            if image_src is None:
                print(f"[Warn] Could not read image: {input_desc}")
                if source["type"] in ["rtsp", "video"]:
                     pass
                continue

            if start_time is None:
                start_time = time.time()
                print(f"[DEBUG] Starting recording for {FRAME_DURATION} seconds...")

            if time.time() - start_time > FRAME_DURATION:
                print(f"[DEBUG] Reaches {FRAME_DURATION}s limit. Stopping.")
                break

            if video_writer is None:
                h, w = image_src.shape[:2]
                out_vid_path = os.path.join(output_dir, "output.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # Use a default FPS of 20 for output, as inference speed varies
                video_writer = cv2.VideoWriter(out_vid_path, fourcc, 20.0, (w, h))
                print(f"[DEBUG] Video writer initialized: {out_vid_path}")
            
            #print(f"[DEBUG] Processing: {input_desc}")
            #print(f"[DEBUG] Image shape: {image_src.shape}")
            image_input, ratio, offset = letter_box(image_src, new_shape=(int(input_size), int(input_size)), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)

            #print("[DEBUG] Running inference...")
            ie_output = ie.run([image_input])
            #print("[DEBUG] Inference done.")
            decoded_tensor = []
            if not final_output in ie.get_output_tensor_names():
                print("[DEBUG] Decoding outputs (multi-layer)...")
                ie_output = [ie_output[i] for i in layer_idx]
                decoded_tensor = all_decode(ie_output, layers, n_classes)
            else:
                #print("[DEBUG] Decoding outputs (single-layer)...")
                decoded_tensor = ie_output[layer_idx[0]]

            #print("[DEBUG] Output decoding done.")

            x = np.squeeze(decoded_tensor)
            #print(f"[DEBUG] Output shape after squeeze: {x.shape}")

            decoding_method = config["model"]["param"].get("decoding_method", "yolo_basic")
            #print(f"[DEBUG] Decoding method: {decoding_method}")

            # Handle transposition if channel-first (common in NPU output for YOLOv8)
            # Only transpose if we have more columns than rows (heuristic for (C, N) vs (N, C))
            # Anchors (N) is typically large (e.g. 8400), Channels (C) is small (e.g. 18 or 85)
            if x.ndim == 2 and x.shape[0] < x.shape[1]:
                #print("[DEBUG] Transposing output tensor...")
                x = x.transpose()
                #print(f"[DEBUG] Output shape after transpose: {x.shape}")

            if decoding_method == "yolov8":
                 # YOLOv8 format: [x, y, w, h, class0, class1, ...]
                 # No objectness score layer
                 box = ops.xywh2xyxy(x[:, :4])
                 cls_scores = x[:, 4:]
                 
                 conf = np.max(cls_scores, axis=1, keepdims=True)
                 j = np.argmax(cls_scores, axis=1, keepdims=True)
                 
                 mask = conf.flatten() > score_threshold
                 filtered = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[mask]

            else:
                 # YOLOv5/Basic format: [x, y, w, h, obj, class0, class1, ...]
                 box = ops.xywh2xyxy(x[:, :4])
                 obj_conf = x[:, 4:5]
                 cls_scores = x[:, 5:]
                 
                 # Combined confidence
                 cls_scores = cls_scores * obj_conf
                 
                 conf = np.max(cls_scores, axis=1, keepdims=True)
                 j = np.argmax(cls_scores, axis=1, keepdims=True)
                 
                 mask = conf.flatten() > score_threshold
                 filtered = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[mask]

            sorted_indices = np.argsort(-filtered[:, 4])
            x = filtered[sorted_indices]
            x = torch.Tensor(x)
            x = x[torchvision.ops.nms(x[:,:4], x[:, 4], score_threshold)]

            #print(f"[DEBUG] Number of boxes after NMS: {len(x)}")
            colors = np.random.randint(0, 256, [n_classes, 3], np.uint8).tolist()
            for idx, r in enumerate(x.numpy()):
                pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
                pt1, pt2 = transform_box(pt1, pt2, ratio, offset, image_src.shape)
                #print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}
                 #     .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
                image_src = cv2.rectangle(image_src, pt1, pt2, colors[label], 2)
                cv2.putText(image_src, '{}: {:.3f}'.format(classes[label], conf), (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
            
            if video_writer is not None:
                video_writer.write(image_src)
            else:
                out_path = os.path.join(output_dir, f"yolov5s_{count}.jpg")
                cv2.imwrite(out_path, image_src)

            #print(f"[DEBUG] Saved file: {out_path}")
            count += 1
        
        if video_writer is not None:
            video_writer.release()
            print("[DEBUG] Video writer released.")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='/home/orangepi/dx-all-suite/dx-runtime/dx_app/example/run_detector/yolov5s1_example.json', type=str, help='yolo object detection json config path')
    parser.add_argument('--output_dir', default='.', type=str, help='directory to save output images')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()

    with open(args.config, "r") as f:
        json_config = json.load(f)
    json_config['output_dir'] = args.output_dir
    run_example(json_config)
