import argparse
import time
from pathlib import Path
import csv  # <-- ★CSV保存のために追加

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import mediapipe as mp  # Mediapipe 
import numpy as np # <-- 顔姿勢推定のために追加 (cv2.solvePnP で使用)

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# --- (EAR関連のヘルパー関数は削除) ---

print("start")

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # --- ★修正: 顔画像保存用のディレクトリ (Pathオブジェクトにし、mkdirを追加) ---
    face_output_dir = Path("/Users/hayatanobuya/1/授業/ハッカソン/detect_yolo/output/images")
    face_output_dir.mkdir(parents=True, exist_ok=True) # ディレクトリが存在しない場合に作成
    # -----------------------------------------------------------------

    # --- ★追加: CSV保存用のディレクトリとファイルパス ---
    csv_output_dir = Path("/Users/hayatanobuya/1/授業/ハッカソン/detect_yolo/output/file")
    csv_output_dir.mkdir(parents=True, exist_ok=True) # ディレクトリを作成
    csv_file_path = csv_output_dir / "results.csv"
    # -----------------------------------------------

    # Initialize
    set_logging()
    device = select_device(opt.device)
    
    # --- Mediapipe  ---
    mp_face_mesh = mp.solutions.face_mesh
    # 
    # refine_landmarks=True を追加して、目と唇のランドマーク精度を向上 (478点)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, 
                                      min_detection_confidence=0.5, 
                                      min_tracking_confidence=0.5, 
                                      refine_landmarks=True) 
    mp_drawing = mp.solutions.drawing_utils
    # --- Mediapipe  ---

    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1


    # --- ★修正: 顔の傾き推定用変数 (カウンター関連を削除) ---
    HEAD_PITCH_THRESH = 20.0 # 前傾（うなずき）のしきい値 (度) (この値は調整が必要)
    # HEAD_PITCH_CONSEC_FRAMES = 15 # (不要になったため削除)
    # FACE_COUNTERS = {} # (不要になったため削除)
    # --- 顔の傾き推定用変数 ---

    # --- solvePnP用 3Dモデル座標 (6点) ---
    # (Y軸は下が負、Z軸は奥が負)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # 1: Nose tip
        (0.0, -330.0, -65.0),        # 152: Chin
        (-225.0, 170.0, -135.0),     # 33: Left eye left corner (左目の内側)
        (225.0, 170.0, -135.0),      # 263: Right eye right corner (右目の内側)
        (-150.0, -150.0, -125.0),    # 61: Left Mouth corner
        (150.0, -150.0, -125.0)      # 291: Right mouth corner
    ], dtype="double")
    # solvePnP用 ランドマークインデックス
    POSE_INDICES = [1, 152, 33, 263, 61, 291]
    # --- solvePnP用 3Dモデル座標 ---

    # --- ★追加: CSVファイルを初期化 (ヘッダー書き込み) ---
    csv_header = ["メッシュID", "パス", "評価"]
    # 'w' (write)モードでファイルを開き、ヘッダーを書き込む
    # 実行のたびに上書きされます
    try:
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(csv_header)
    except Exception as e:
        print(f"Error initializing CSV file {csv_file_path}: {e}")
    # ----------------------------------------------------


    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                        # --- Mediapipe  ---
                        #  (int(cls) == 0) (person クラスを想定)
                        if int(cls) == 0:
                            try:
                                # YOLOv7
                                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                                
                                # 
                                if x1 < 0: x1 = 0
                                if y1 < 0: y1 = 0
                                if x2 > im0.shape[1]: x2 = im0.shape[1]
                                if y2 > im0.shape[0]: y2 = im0.shape[0]
                                
                                # 
                                person_roi = im0[y1:y2, x1:x2]
                                roi_shape = person_roi.shape # (height, width, channels)

                                # 
                                if person_roi.size == 0:
                                    continue # 

                                # OpenCV(BGR)  RGB 
                                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                                roi_rgb.flags.writeable = False # 
                                
                                # Mediapipe 
                                results = face_mesh.process(roi_rgb)
                                
                                roi_rgb.flags.writeable = True # 

                                # 
                                if results.multi_face_landmarks:
                                    
                                    for face_landmarks in results.multi_face_landmarks:
                                        
                                        # (簡易ID: 最初のランドマークのy座標)
                                        face_id = int(face_landmarks.landmark[0].y * 100) 
                                        print(f"Detected Face ID: {face_id}") # コンソール出力

                                        # --- ★修正: CSV書き込み用変数の初期化 ---
                                        face_save_path_str = None # CSVに書き込むパス
                                        status = "unknown" # 評価のデフォルト
                                        color = (255, 0, 0) # デフォルト色 (青)
                                        # ---------------------------------------

                                        # --- ★修正: 顔画像の保存処理 (サブディレクトリなし) ---
                                        if opt.save_faces: # --save-faces が有効な場合のみ保存
                                            
                                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                                            # ★修正: face_id をファイル名に含めて一意にする
                                            face_filename = f"face_id_{face_id}_{timestamp}_{frame}.jpg"
                                            
                                            # ★修正: face_output_dir (ベースパス) とファイル名を結合
                                            face_save_path = face_output_dir / face_filename
                                            face_save_path_str = str(face_save_path) # ★CSV用に文字列パスを保持
                                            
                                            # ★修正: cv2.imwrite に「完全なファイルパス(str)」と「画像(person_roi)」を渡す
                                            # person_roi は Mediapipe 処理前の元の顔画像 (YOLOv7で検出された範囲)
                                            cv2.imwrite(str(face_save_path), person_roi) 
                                            print(f"Saved face for ID {face_id} to {face_save_path}")
                                        # ----------------------------------------
                                        
                                        # ★修正: FACE_COUNTERS への格納処理を削除 (不要)

                                        all_landmarks = face_landmarks.landmark
                                        
                                        
                                        # --- 顔の傾き推定 (solvePnP) ---
                                        try:
                                            # 1. カメラ行列の準備 (簡易版)
                                            focal_length = roi_shape[1] # width
                                            center = (roi_shape[1] / 2, roi_shape[0] / 2) # (width/2, height/2)
                                            camera_matrix = np.array(
                                                [[focal_length, 0, center[0]],
                                                 [0, focal_length, center[1]],
                                                 [0, 0, 1]], dtype="double"
                                            )
                                            dist_coeffs = np.zeros((4, 1)) # 歪みなしと仮定

                                            # 2. 2D画像座標の取得 (ROI座標系)
                                            image_points = []
                                            for idx in POSE_INDICES:
                                                lm = all_landmarks[idx]
                                                cx = int(lm.x * roi_shape[1]) # roi width
                                                cy = int(lm.y * roi_shape[0]) # roi height
                                                image_points.append((cx, cy))
                                            image_points = np.array(image_points, dtype="double")
                                            
                                            # 3. solvePnP 実行
                                            (success, rotation_vector, translation_vector) = cv2.solvePnP(
                                                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                                            )

                                            # 4. 回転ベクトルをオイラー角 (Pitch, Yaw, Roll) に変換
                                            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                                            sy = np.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
                                            
                                            singular = sy < 1e-6
                                            
                                            if not singular:
                                                pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
                                                yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
                                                roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
                                            else:
                                                pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
                                                yaw = 0
                                                roll = np.degrees(np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]))

                                            # 5. 状態判定 (Pitch: 前後の傾き)
                                            # pitch > 0 がうなずき (前傾)
                                            
                                            # --- ★修正: 評価ロジックを "good" / "bad" に変更 ---
                                            if pitch > HEAD_PITCH_THRESH:
                                                status = "bad" # ★ここで status 更新
                                                color = (0, 255, 0) # 緑
                                            else:
                                                status = "good" # ★ここで status 更新
                                                color = (0, 0, 255) # 赤
                                            
                                            # ★追加: 評価をコンソールに出力
                                            print(f"  Face ID {face_id} Pitch: {pitch:.1f} deg -> Status: {status}")
                                            # ---------------------------------------------------

                                            # 6. 結果の描画 (ROI座標系)
                                            # status = FACE_COUNTERS[face_id]['status'] # (旧ロジック削除)
                                            if status:
                                                # ROI の左上に描画 (色も変更)
                                                cv2.putText(person_roi, status, (10, 30), 
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                                            # Pitch値のデバッグ描画
                                            cv2.putText(person_roi, f"Pitch: {pitch:.1f} deg", (10, 60), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                                        except Exception as e_pose:
                                            print(f"Error in Head Pose estimation: {e_pose}")
                                            status = "error" # ★ポーズ推定エラーの場合
                                        # --- 顔の傾き推定 終了 ---

                                        # --- ★追加: CSVファイルへの追記 ---
                                        # --save-faces が指定されている場合のみ、CSVに書き込む
                                        # (画像パス face_save_path_str が "パス" 列に必要なため)
                                        if opt.save_faces:
                                            try:
                                                # 'a' (append)モードでファイルを開く
                                                with open(csv_file_path, 'a', newline='', encoding='utf-8') as f_csv:
                                                    writer = csv.writer(f_csv)
                                                    # [メッシュID, パス, 評価]
                                                    row_data = [face_id, face_filename, status]
                                                    writer.writerow(row_data)
                                            except Exception as e:
                                                print(f"Error writing to CSV file: {e}")
                                        # -----------------------------------


                                        # --- 顔メッシュ描画 (元のコードの cv2.circle の代わり) ---
                                        # メッシュを ROI (person_roi) に描画
                                        mp_drawing.draw_landmarks(
                                            image=person_roi, # ROI 
                                            landmark_list=face_landmarks,
                                            connections=mp_face_mesh.FACEMESH_TESSELATION, 
                                            landmark_drawing_spec=None,
                                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
                                        
                                        # 状態を描画した person_roi を im0 に戻す
                                        im0[y1:y2, x1:x2] = person_roi

                                        # 1つの顔だけ処理して break
                                        break 
                                        
                            except Exception as e:
                                print(f"Error processing face mesh: {e}")
                        # --- Mediapipe  ---


            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")
    
    # ★CSVファイルの保存場所をコンソールに表示
    if opt.save_faces:
        print(f"CSV results saved to {csv_file_path}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    # --- ★追加: 顔画像の保存を制御する引数 ---
    # ★ (この引数がCSV保存も兼ねます)
    parser.add_argument('--save-faces', action='store_true', help='save detected face images and CSV results by Face ID')
    # ----------------------------------------
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()