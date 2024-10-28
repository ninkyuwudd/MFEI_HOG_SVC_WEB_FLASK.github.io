
import cv2
import os


def create_video_from_frames(frame_folder, output_video_path, fps=25):
    # Dapatkan daftar file frame dan urutkan
    frames = [f for f in os.listdir(frame_folder) if f.endswith('.png') or f.endswith('.jpg')]
    frames.sort()

    # Baca frame pertama untuk mendapatkan ukuran video
    first_frame_path = os.path.join(frame_folder, frames[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape

    # Tentukan codec dan buat VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec untuk format .mp4
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Tulis setiap frame ke video
    for frame_name in frames:
        frame_path = os.path.join(frame_folder, frame_name)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Selesai menulis video
    video_writer.release()
    print(f"Video saved to {output_video_path}")


def extract_frames_from_video(video_path, output_folder):
    # Buat folder output jika belum ada

    os.makedirs(output_folder, exist_ok=True)

    # Buka video
    video_capture = cv2.VideoCapture(video_path)

    # Periksa apakah video berhasil dibuka
    if not video_capture.isOpened():
        message = f"Error: Tidak dapat membuka video {video_path}"
        return message

    frame_count = 0
    while True:
        # Baca frame dari video
        ret, frame = video_capture.read()

        # Jika tidak ada frame lagi, keluar dari loop
        if not ret:
            break

        # Simpan frame sebagai gambar
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Selesai memproses video
    video_capture.release()
    return f"Extracted {frame_count} frames to {output_folder}"


def check_input_video(request,app):
    if request.method == "POST":
        if "videoInput" not in request.files:
            return 'Is empty file'

        dataVideo = request.files['videoInput']

        if dataVideo.filename == "":
            return 'You not select any video'
        
        if dataVideo:

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], dataVideo.filename)
            dataVideo.save(file_path)
            result = extract_frames_from_video(file_path, "output_frame")
            return f'File  {file_path} and {result}'

# Contoh penggunaan
frame_folder = 'E:/kuliah/skripsi/datavideo/casia/fn/fn00/0000'  # Ganti dengan path ke folder frame Anda
output_video_path = 'output_video.mp4'
# create_video_from_frames(frame_folder, output_video_path)

