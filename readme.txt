# 1. Ensure the camera node is functioning correctly (new terminal startup).
source /opt/ros/noetic/setup.bash
roscore

# 2. Start usb_cam (valid parameters)
source /opt/ros/noetic/setup.bash
rosrun usb_cam usb_cam_node _video_device:=/dev/video4 _image_width:=640 _image_height:=480 _pixel_format:=yuyv

# 3. Run the inference script
cd ~/onnx_ros_ws/src/onnx_camera_recognition
source /opt/ros/noetic/setup.bash
source ~/onnx_ros_ws/devel/setup.bash
python3 onnx_inference_nodev5.py


/home/mustar/miniconda3/envs/yolov11/bin/python