from djitellopy import Tello
import config

# CREATE TELLO
def create_tello():
    tello = Tello()
    tello.connect()
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0
    return tello


# CONNECT TO TELLO
def connect_tello(tello):
    tello.connect()
    tello.for_back_velocity = 0
    tello.left_right_velocity = 0
    tello.up_down_velocity = 0
    tello.yaw_velocity = 0
    tello.speed = 0

def get_tello_image(tello):
    # GET THE IMAGE FROM TELLO
    frame_read = tello.get_frame_read()
    print("get frame")
    my_frame = frame_read.frame
    return my_frame

# def check_frame():
#     # Check if camera opened successfully
#     if (cap.isOpened()== False):  #change to frame###
#       print("Error opening video stream or file")

