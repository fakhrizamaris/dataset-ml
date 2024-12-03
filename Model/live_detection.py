import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load model yang sudah dilatih
model = tf.keras.models.load_model('best_model2.keras')

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Konversi frame ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi landmark wajah
    result = face_mesh.process(rgb_frame)
    
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Ekstrak landmark menjadi array
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark]).flatten()
            
            # Reshape sesuai input model
            landmarks = landmarks.reshape(1, -1)
            
            # Prediksi bentuk wajah
            prediction = model.predict(landmarks)
            face_shape = np.argmax(prediction)  # Mengambil kelas dengan probabilitas tertinggi
            
            # Tampilkan prediksi pada frame
            cv2.putText(frame, f'Face Shape: {face_shape}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Tampilkan frame
    cv2.imshow('Face Shape Detection', frame)
    
    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
