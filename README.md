## 🔥🔥This is our project for OUC and Heriot Watt course IRP2

 Our project is a automatic ping pang ball pick up robot based on DJI EP robot and Elephant Arm

The main function is the detect.py in yolo_utils and the movement logic is in botMove.py

If you use our code, please star our repo.
## ✉✉ Contact
If you have any questions, do not hesitate to contact us!
Email: [xiejiangwei@stu.ouc.edu.cn)(xiejiangwei@stu.ouc.edu.cn)

## Introduction
Make sure your group has ```2 PC devices```, one for connecting DJI robot and the other one for connecting elephant arm through **SSH** or **VNC**
### DJI device
We have upload the pretrained model checkpoint for recognising ping pang ball, please revise the checkpoint path in ```detect.py``` then your DJI robot could recognise ping pang ball and move.
For distance judjing, there is a threshold value for detecting the square of the yolo box, maybe you need to adjust this value for your device.
![10a2cf10f69dd2ec390ab486be93551](https://github.com/user-attachments/assets/458860fe-48da-4a7b-9c62-6643b86837b7, =300x200)

### Elephant Arm device
Start ```video.py``` file as long as another device programms run, the robot arm camera will check whether there is a ball in its view through cv2 method.
