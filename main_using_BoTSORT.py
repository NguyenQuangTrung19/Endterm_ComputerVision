# Các thư viện cần thiết
import argparse
from enum import Enum
from typing import Iterator, List
import sys
import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BOTSORT_PATH = os.path.join(SCRIPT_DIR, "BoT-SORT")

sys.path.insert(0, BOTSORT_PATH)
from tools import track  
from tracker.bot_sort import BoTSORT

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Các hằng số
PLAYER_DETECTION_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/yolov8x_train_player.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/yolov8x-pose_train_pitch.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(SCRIPT_DIR, 'model/yolov8x_train_ball.pt')

REID_MODEL_PATH = os.path.join(BOTSORT_PATH, "weights", "osnet_x0_25_msmt17.pth")

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(color=[sv.Color.from_hex(c) for c in CONFIG.colors], text_color=sv.Color.from_hex('#FFFFFF'), border_radius=5)
EDGE_ANNOTATOR = sv.EdgeAnnotator(color=sv.Color.from_hex('#FF1493'), thickness=2, edges=CONFIG.edges)
BOX_ANNOTATOR = sv.BoxAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(color=sv.ColorPalette.from_hex(COLORS), thickness=2)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(COLORS), text_color=sv.Color.from_hex('#FFFFFF'), text_position=sv.Position.BOTTOM_CENTER)

# Định nghĩa một tập hợp các chế độ hoạt động cố định cho script.
class Mode(Enum):
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'

# Hàm cắt và trả về một danh sách các ảnh nhỏ (crops) của các đối tượng
def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(players: sv.Detections, players_team_id: np.ndarray, goalkeepers: sv.Detections) -> np.ndarray:
    '''
        Hàm giúp xác định đội cho thủ môn. Vì thủ môn thường mặc áo khác màu, 
        hàm này sẽ gán thủ môn vào đội có các cầu thủ ở gần anh ta nhất.
    '''
    if len(players) == 0 or len(players_team_id) == 0 or len(goalkeepers) == 0:
        return np.array([-1] * len(goalkeepers))
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_players = players_xy[players_team_id == 0]
    team_1_players = players_xy[players_team_id == 1]
    if len(team_0_players) == 0 or len(team_1_players) == 0:
         return np.array([-1] * len(goalkeepers))
    team_0_centroid = team_0_players.mean(axis=0)
    team_1_centroid = team_1_players.mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def render_radar(detections: sv.Detections, keypoints: sv.KeyPoints, color_lookup: np.ndarray) -> np.ndarray:
    '''
        Hàm tạo ra một hình ảnh sân bóng 2D từ trên cao (radar view). 
        Nó sử dụng phép biến đổi phối cảnh (perspective transform) để chiếu vị trí của các cầu thủ từ khung hình 
        video lên bản đồ sân bóng, giúp người xem có cái nhìn tổng quan về chiến thuật.
    '''
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not np.any(mask):
        return draw_pitch(config=CONFIG)
    transformer = ViewTransformer(source=keypoints.xy[0][mask].astype(np.float32), target=np.array(CONFIG.vertices)[mask].astype(np.float32))
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)
    radar = draw_pitch(config=CONFIG)
    for i in range(len(COLORS)):
        if np.any(color_lookup == i):
            radar = draw_points_on_pitch(config=CONFIG, xy=transformed_xy[color_lookup == i], face_color=sv.Color.from_hex(COLORS[i]), radius=20, pitch=radar)
    return radar

def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    '''
        Hàm chạy model phát hiện sân bóng, vẽ các điểm keypoint 
        và nhãn của chúng lên video để cho thấy các góc và đường kẻ sân đã được nhận dạng
    '''
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame

def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
        Hàm chạy model phát hiện cầu thủ, Vẽ các bounding box xung quanh cầu thủ, thủ môn và trọng tài được phát hiện.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame

def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
        Chạy model phát hiện bóng, sử dụng BallTracker để làm mượt quỹ đạo của bóng 
        và BallAnnotator để vẽ vị trí bóng lên video.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    slicer = sv.InferenceSlicer(callback=callback, slice_wh=(640, 640))
    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame

def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
        Hàm kết hợp phát hiện cầu thủ (YOLO) và thuật toán theo dõi BoT-SORT.
        Gán một ID duy nhất (tracker_id) cho mỗi cầu thủ và duy trì ID đó qua các khung hình.
        Vẽ hình elip và ID của từng cầu thủ lên video.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    parser = track.make_parser()
    default_args = parser.parse_args(['dummy.mp4'])
    default_args.device = device
    default_args.fp16 = False if device == 'cpu' else True
    default_args.with_reid = True
    default_args.fast_reid_weights = REID_MODEL_PATH
    default_args.fast_reid_config = os.path.join(BOTSORT_PATH, "fast_reid/configs/MOT17/sbs_S50.yml")
    default_args.name = ''
    default_args.ablation = False
    default_args.cmc_method = "sparseOptFlow"
    tracker = BoTSORT(args=default_args)

    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections_yolo = sv.Detections.from_ultralytics(result)
        
        dets_for_botsort = np.array([[*box, conf, cls] for box, conf, cls in zip(detections_yolo.xyxy, detections_yolo.confidence, detections_yolo.class_id)])
        if dets_for_botsort.shape[0] > 0:
            online_targets = tracker.update(dets_for_botsort, frame)
            detections = sv.Detections(
                xyxy=np.array([t.tlbr for t in online_targets]),
                tracker_id=np.array([t.track_id for t in online_targets]).astype(int),
                class_id=np.array([t.cls for t in online_targets]).astype(int)
            ) if online_targets else sv.Detections.empty()
        else:
            detections = sv.Detections.empty()

        labels = [f"ID:{tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []
        annotated_frame = frame.copy()
        if len(detections) > 0:
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections, labels=labels)
        
        yield annotated_frame

def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
        Hàm này mở rộng chức năng PLAYER_TRACKING.
        Bước 1: Lướt qua video một lần để thu thập các ảnh cắt (crops) của cầu thủ.
        Bước 2: "Huấn luyện" một bộ phân loại (TeamClassifier) trên các ảnh cắt này 
        để học cách phân biệt hai đội (thường dựa vào màu áo).
        Bước 3: Chạy lại video, vừa theo dõi cầu thủ, vừa dùng bộ phân loại đã huấn 
        luyện để gán đội (Team 0 hoặc Team 1) cho mỗi người và tô màu tương ứng.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)

    frame_generator_for_crops = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator_for_crops, desc='Collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        crops.extend(get_crops(frame, player_detections))
    if crops:
        team_classifier.fit(crops)

    parser = track.make_parser()
    default_args = parser.parse_args(['dummy.mp4'])
    default_args.device = device
    default_args.fp16 = False if device == 'cpu' else True
    default_args.with_reid = True
    default_args.fast_reid_weights = REID_MODEL_PATH
    default_args.fast_reid_config = os.path.join(BOTSORT_PATH, "fast_reid/configs/MOT17/sbs_S50.yml")
    default_args.name = ''
    default_args.ablation = False
    default_args.cmc_method = "sparseOptFlow"
    tracker = BoTSORT(args=default_args)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result_yolo = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections_yolo = sv.Detections.from_ultralytics(result_yolo)
        
        dets_for_botsort = np.array([[*box, conf, cls] for box, conf, cls in zip(detections_yolo.xyxy, detections_yolo.confidence, detections_yolo.class_id)])
        if dets_for_botsort.shape[0] > 0:
            online_targets = tracker.update(dets_for_botsort, frame)
            detections = sv.Detections(
                xyxy=np.array([t.tlbr for t in online_targets]),
                tracker_id=np.array([t.track_id for t in online_targets]).astype(int),
                class_id=np.array([t.cls for t in online_targets]).astype(int)
            ) if online_targets else sv.Detections.empty()
        else:
            detections = sv.Detections.empty()

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        players_team_id = np.array([])
        if len(players) > 0 and team_classifier.fitted:
            player_crops = get_crops(frame, players)
            if player_crops: players_team_id = team_classifier.predict(player_crops)
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers) if len(players_team_id) == len(players) else np.array([])

        detections_list, color_lookup_list = [], []
        if len(players) > 0:
            detections_list.append(players)
            color_lookup_list.extend(players_team_id.tolist() if len(players_team_id) == len(players) else [PLAYER_CLASS_ID] * len(players))
        if len(goalkeepers) > 0:
            detections_list.append(goalkeepers)
            color_lookup_list.extend(goalkeepers_team_id.tolist() if len(goalkeepers_team_id) == len(goalkeepers) else [GOALKEEPER_CLASS_ID] * len(goalkeepers))
        if len(referees) > 0:
            detections_list.append(referees)
            color_lookup_list.extend([REFEREE_CLASS_ID] * len(referees))

        annotated_frame = frame.copy()
        if detections_list:
            detections_for_annotation = sv.Detections.merge(detections_list)
            color_lookup = np.array(color_lookup_list)
            labels = [f"ID:{tracker_id}" for tracker_id in detections_for_annotation.tracker_id]
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections_for_annotation, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections_for_annotation, labels, custom_color_lookup=color_lookup)
        yield annotated_frame

def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    '''
        Chức năng toàn diện, kết hợp tất cả các chức năng: phát hiện sân, theo dõi cầu thủ, và phân loại đội.
    '''
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)
    
    frame_generator_for_crops = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator_for_crops, desc='Collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops.extend(get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID]))
    if crops:
        team_classifier.fit(crops)

    parser = track.make_parser()
    default_args = parser.parse_args(['dummy.mp4'])
    default_args.device = device
    default_args.fp16 = False if device == 'cpu' else True
    default_args.with_reid = True
    default_args.fast_reid_weights = REID_MODEL_PATH
    default_args.fast_reid_config = os.path.join(BOTSORT_PATH, "fast_reid/configs/MOT17/sbs_S50.yml")
    default_args.name = ''
    default_args.ablation = False
    default_args.cmc_method = "sparseOptFlow"
    tracker = BoTSORT(args=default_args)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result_pitch = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result_pitch)
        result_yolo = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections_yolo = sv.Detections.from_ultralytics(result_yolo)
        
        dets_for_botsort = np.array([[*box, conf, cls] for box, conf, cls in zip(detections_yolo.xyxy, detections_yolo.confidence, detections_yolo.class_id)])
        if dets_for_botsort.shape[0] > 0:
            online_targets = tracker.update(dets_for_botsort, frame)
            detections = sv.Detections(
                xyxy=np.array([t.tlbr for t in online_targets]),
                tracker_id=np.array([t.track_id for t in online_targets]).astype(int),
                class_id=np.array([t.cls for t in online_targets]).astype(int)
            ) if online_targets else sv.Detections.empty()
        else:
            detections = sv.Detections.empty()

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        players_team_id = np.array([])
        if len(players) > 0 and team_classifier.fitted:
            player_crops = get_crops(frame, players)
            if player_crops: players_team_id = team_classifier.predict(player_crops)
        goalkeepers_team_id = resolve_goalkeepers_team_id(players, players_team_id, goalkeepers) if len(players_team_id) == len(players) else np.array([])
        
        detections_list, color_lookup_list = [], []
        if len(players) > 0:
            detections_list.append(players)
            color_lookup_list.extend(players_team_id.tolist() if len(players_team_id) == len(players) else [PLAYER_CLASS_ID] * len(players))
        if len(goalkeepers) > 0:
            detections_list.append(goalkeepers)
            color_lookup_list.extend(goalkeepers_team_id.tolist() if len(goalkeepers_team_id) == len(goalkeepers) else [GOALKEEPER_CLASS_ID] * len(goalkeepers))
        if len(referees) > 0:
            detections_list.append(referees)
            color_lookup_list.extend([REFEREE_CLASS_ID] * len(referees))

        annotated_frame = frame.copy()
        if detections_list:
            detections_for_annotation = sv.Detections.merge(detections_list)
            color_lookup = np.array(color_lookup_list)
            labels = [f"ID:{tracker_id}" for tracker_id in detections_for_annotation.tracker_id]
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections_for_annotation, custom_color_lookup=color_lookup)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(annotated_frame, detections_for_annotation, labels, custom_color_lookup=color_lookup)
            
            h, w, _ = frame.shape
            radar_img = render_radar(detections_for_annotation, keypoints, color_lookup)
            radar_img = sv.resize_image(radar_img, (w // 3, h // 3))
            radar_h, radar_w, _ = radar_img.shape
            rect = sv.Rect(x=w - radar_w, y=h - radar_h, width=radar_w, height=radar_h)
            annotated_frame = sv.draw_image(annotated_frame, radar_img, opacity=0.8, rect=rect)

        yield annotated_frame

# Hàm thực thi chính
def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:
    if not os.path.exists(REID_MODEL_PATH):
        print(f"LỖI: Không tìm thấy file trọng số Re-ID tại: {REID_MODEL_PATH}")
        print("Vui lòng tải file 'osnet_x0_25_msmt17.pth' và đặt vào thư mục 'BoT-SORT/weights/'.")
        return
    if not os.path.exists(PLAYER_DETECTION_MODEL_PATH) or not os.path.exists(PITCH_DETECTION_MODEL_PATH):
        print("LỖI: Không tìm thấy file model của YOLO. Hãy chắc chắn các file .pt nằm trong thư mục 'model/'.")
        return

    mode_to_function = {
        Mode.PITCH_DETECTION: run_pitch_detection,
        Mode.PLAYER_DETECTION: run_player_detection,
        Mode.BALL_DETECTION: run_ball_detection,
        Mode.PLAYER_TRACKING: run_player_tracking,
        Mode.TEAM_CLASSIFICATION: run_team_classification,
        Mode.RADAR: run_radar
    }
    if mode not in mode_to_function:
        raise NotImplementedError(f"Mode {mode} is not implemented.")
    frame_generator = mode_to_function[mode](source_video_path=source_video_path, device=device)
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        with tqdm(total=video_info.total_frames, desc=f"Processing video in {mode.value} mode") as pbar:
            for frame in frame_generator:
                sink.write_frame(frame)
                
                display_frame = sv.resize_image(image=frame, resolution_wh=(1280, 720))
                cv2.imshow("Soccer-CV", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                pbar.update(1)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phân tích video bóng đá.')
    parser.add_argument('--source_video_path', type=str, required=True, help='Đường dẫn đến video nguồn.')
    parser.add_argument('--target_video_path', type=str, required=True, help='Đường dẫn để lưu video kết quả.')
    parser.add_argument('--device', type=str, default='cpu', help="Thiết bị để chạy model ('cpu' or 'cuda').")
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_TRACKING, choices=list(Mode), help='Chế độ phân tích để chạy.')
    args = parser.parse_args()
    
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )