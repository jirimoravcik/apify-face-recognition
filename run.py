import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
from dotenv import load_dotenv

load_dotenv()
IDENTITY_VECTORS_FOLDER_NAME = os.environ.get("IDENTITY_VECTORS_FOLDER_NAME")

face_app = FaceAnalysis(
    name="buffalo_l", providers=["CoreMLExecutionProvider", "CPUExecutionProvider"]
)
face_app.prepare(ctx_id=0)


def load_identity_database(folder):
    idx = 0
    idx_to_name = {}
    embeddings = []
    for file in os.listdir(folder):
        embeddings.append(np.load(os.path.join(folder, file)))
        idx_to_name[idx] = file.split(".")[0]
        idx += 1
    return np.vstack(embeddings), idx_to_name


def detect_from_video_capture():
    orig_embeddings, idx_to_name = load_identity_database(IDENTITY_VECTORS_FOLDER_NAME)
    stream = cv2.VideoCapture(0)

    while True:
        (grabbed, frame) = stream.read()

        if not grabbed:
            break

        faces = face_app.get(frame)
        for face in faces:
            embedding = face.normed_embedding
            scores = orig_embeddings @ embedding[..., np.newaxis]
            name = idx_to_name.get(
                np.argmax(scores), "unknown"
            )  # + f', age: {face.age}'
            left, top, width, height = (
                int(face.bbox[0]),
                int(face.bbox[1]),
                int(face.bbox[2] - face.bbox[0]),
                int(face.bbox[3] - face.bbox[1]),
            )
            cv2.rectangle(
                frame, (left, top, width, height), color=(0, 0, 255), thickness=2
            )
            cv2.putText(
                frame,
                name,
                (left - 5, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()

    stream.release()


if __name__ == "__main__":
    detect_from_video_capture()
