# IMPORTS:
import cv2
import mediapipe as mp

# GET THE NECESSARY RESOURCES:
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# FOR STATIC IMAGES:
IMAGE_FILES = []
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)

with mp_face_mesh.FaceMesh (
    static_image_mode = True,
    max_num_faces = 1,
    refine_landmarks=  True,
    min_detection_confidence = 0.5) as face_mesh:
 
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)

    # CONVERT THE BGR IMAGE TO RGB BEFORE PROCESSING:
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # PRINT AND DRAW FACE MESH LANDMARKS ON THE IMAGE:
    if not results.multi_face_landmarks:
      continue
    annotated_image = image.copy()

    # LANDMARKS:
    for face_landmarks in results.multi_face_landmarks:
      print('face_landmarks:', face_landmarks)
      mp_drawing.draw_landmarks(
          image=annotated_image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())

    # DRAW LANDMARKS ON DETECTED FACES:
      mp_drawing.draw_landmarks(
          image = annotated_image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_CONTOURS,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())

    # DRAW LANDMARKS ON DETECTED IRIS'S:
      mp_drawing.draw_landmarks(
          image = annotated_image,
          landmark_list = face_landmarks,
          connections = mp_face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec = None,
          connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # DRAW IMAGE:
    cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# FOR WEBCAM INPUT:
drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # IF LOADING A VIDEO, USE 'BREAK' INSTEAD OF 'CONTINUE':
      continue

    # TO IMPROVE PERFORMANCE, OPTIONALLY MARK THE IMAGE AS NOT WRITEABLE TO,
    # PASS BY REFERENCE.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # R.G.B colour.
    results = face_mesh.process(image) # Process the mesh image with the set colour.

    # DRAW THE FACE MESH ANNOTATIONS ON THE IMAGE:
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # MULTIPLE LANDMARKS (TESSELATION) ON FACES:
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # DRAW FACE MESH CONTOUR STYLE LANDMARKS:
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())

        # DRAW FACE MESH IRIS CONNECTION LANDMARKS:
        mp_drawing.draw_landmarks(
            image = image,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style())

    # FLIP THE IMAGE HORIZONTALLY FOR A SELFIE-VIEW DISPLAY:
    cv2.imshow('Samaritan Face Mesh Recognition Module', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release() # Release the video capture.