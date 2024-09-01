import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
from pynput.keyboard import Controller
import pyautogui


pyautogui.hotkey('win', 'ctrl', 'o')  # Atalho para abrir o teclado virtual

cap = cv2.VideoCapture(0)
board = np.zeros((600, 1400), np.uint8)
board[:] = 255

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
font = cv2.FONT_HERSHEY_SIMPLEX

# Keyboard settings
keyboard = np.zeros((300, 700, 3), np.uint8)  # Reduzido para metade do tamanho
keys_set = {
    0: "a", 1: "b", 2: "c", 3: "d", 4: "e",
    5: "f", 6: "g", 7: "h", 8: "i", 9: "j",
    10: "k", 11: "l", 12: "m", 13: "n", 14: "o",
    15: "p", 16: "q", 17: "r", 18: "s", 19: "t",
    20: "u", 21: "v", 22: "w", 23: "x", 24: "y",
    25: "z", 26: "_", 27: "WIN", 28: "del", 29: "F5"
}

def draw_letters(letter_index, text, letter_light):
    x = (letter_index % 10) * 70   # Reduzido para metade
    y = (letter_index // 10) * 100 # Reduzido para metade

    width = 70   # Reduzido para metade
    height = 100 # Reduzido para metade
    th = 2  # thickness reduzido

    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 2.5  # Reduzido para metade
    font_th = 1       # Reduzido
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]
    text_x = int((width - width_text) / 2) + x
    text_y = int((height + height_text) / 2) + y

    if letter_light:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (51, 51, 51), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def get_gaze_direction(facial_landmarks):
    # Calcular o centro dos olhos
    left_eye_center = midpoint(facial_landmarks.part(36), facial_landmarks.part(39))
    right_eye_center = midpoint(facial_landmarks.part(42), facial_landmarks.part(45))

    eye_direction = (right_eye_center[0] + left_eye_center[0]) // 2, (right_eye_center[1] + left_eye_center[1]) // 2

    # Ajustar a sensibilidade
    sensitivity = 10  # Aumentar para maior sensibilidade

    keyboard_center_x = keyboard.shape[1] // 2
    keyboard_center_y = keyboard.shape[0] // 2

    eye_direction_mapped_x = np.interp(eye_direction[0], [0, cols], [0, keyboard.shape[1] - 1])
    eye_direction_mapped_y = np.interp(eye_direction[1], [0, rows], [0, keyboard.shape[0] - 1])

    # Inverter o eixo X para corrigir a  do movimento
    eye_direction_mapped_x = keyboard.shape[1] - eye_direction_mapped_x

    # Ajustar a sensibilidade ao movimento ocular em torno do centro do teclado
    eye_direction_mapped_x = (eye_direction_mapped_x - keyboard_center_x) * sensitivity + keyboard_center_x
    eye_direction_mapped_y = (eye_direction_mapped_y - keyboard_center_y) * sensitivity + keyboard_center_y

    # Garantir que o cursor esteja dentro dos limites do teclado
    eye_direction_mapped_x = np.clip(eye_direction_mapped_x, 0, keyboard.shape[1] - 1)
    eye_direction_mapped_y = np.clip(eye_direction_mapped_y, 0, keyboard.shape[0] - 1)

    return int(eye_direction_mapped_x), int(eye_direction_mapped_y)



# Counters and settings
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_select_letter = 50  # 2-3 seconds
quick_blink_frames = 5
long_blink_frames = 30  # Adjusted for longer blink
text = ""



# Initialize pynput keyboard controller
keyboard_controller = Controller()


while True:
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        left_eye, right_eye = eyes_contour_points(landmarks)
        gaze_direction = get_gaze_direction(landmarks)

        # Detect blinking
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)

        if blinking_ratio > 5:  # Blink detected
            blinking_frames += 1

            # Long blink selects letter
            if blinking_frames >= long_blink_frames:
                x, y = gaze_direction
                letter_index = (y // 100) * 10 + (x // 70)  # Ajustado para as novas 
                active_letter = keys_set[letter_index]
                

                if active_letter != "_":
                    
                    if active_letter == "WIN":
                        keyboard_controller.Key.cmd
                        keyboard_controller.Key.cmd
                    elif active_letter == "F5":
                        keyboard_controller.Key.f5
                        keyboard_controller.Key.f5
                    elif active_letter == "delete":  
                        pyautogui.press('backspace')
                        text = text[:-1]  # Remove o  caractere do texto
                    else:
                        keyboard_controller.press(active_letter)  # Simula a tecla pressionada
                        keyboard_controller.release(active_letter)
                        text += active_letter

                else:
                    text += " "
                    keyboard_controller.press(" ")
                    keyboard_controller.release(" ")
                blinking_frames = 0
        else:
            blinking_frames = 0

    # Primeiro, desenhe as letras
    for i in range(len(keys_set)):
        draw_letters(i, keys_set[i], i == letter_index)

    # Em seguida, desenhe o cursor
    cv2.circle(keyboard, gaze_direction, 10, (0, 255, 0), 2)  # Ajustado o  para tamanho menor

    # Mostrar o texto que estamos escrevendo no quadro
    cv2.putText(board, text, (40, 100), font, 4.5, (0, 0, 0), 2)  # Reduzido para metade

    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    if key == 27:
        #apague a ultima letra escrita no array
        erase_letters(i, key_set[i], draw_letters[-1])

        break

cap.release()
cv2.destroyAllWindows()


