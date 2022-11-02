import cv2

video_path = '20221101_202152.mp4'

video = cv2.VideoCapture(video_path)
    
ret = True

if not video.isOpened():
    print('Error opening video')
    exit(69)

num = 0

while ret:
    ret, frame = video.read()
    cv2.imwrite(f'{num}.jpg', frame)
    num += 1