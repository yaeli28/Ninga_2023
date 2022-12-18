import connection
def set_velocity(self,tello,dir):
    if dir == 1:
        tello.yaw_velocity = -60
    elif dir == 2:
        tello.yaw_velocity = 60
    elif dir == 3:
        tello.up_down_velocity = 60
    elif dir == 4:
        tello.up_down_velocity = -60
    else:
        tello.left_right_velocity = 0;
        tello.for_back_velocity = 0;
        tello.up_down_velocity = 0;
        tello.yaw_velocity = 0
        # SEND VELOCITY VALUES TO TELLO
    if tello.send_rc_control:
        tello.send_rc_control(tello.left_right_velocity, tello.for_back_velocity, tello.up_down_velocity, tello.yaw_velocity)
    print(dir)







