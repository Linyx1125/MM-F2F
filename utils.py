import numpy as np
from batch_face import RetinaFace


detector = RetinaFace(gpu_id=0)

def detect_faces(frames):
    if len(np.array(frames).shape) == 3:
        frames = [frames]
    all_faces = detector(frames)
    face_frames = []
    for (faces, frame) in zip(all_faces, frames):
        if len(faces) == 0:
            return None
        bbox, landmarks, score = faces[0]
        x1, y1, x2, y2 = list(map(int, bbox))
        face_frame = frame[y1:y2, x1:x2, ::-1]
        face_frames.append(face_frame)
    return face_frames