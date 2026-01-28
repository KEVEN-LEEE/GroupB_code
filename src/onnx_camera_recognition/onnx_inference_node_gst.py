#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import onnxruntime as ort
import subprocess
import os
import sys
import threading
import time # import time

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String, Bool

# --- Configuration ---
SPEECH_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_interaction_ros_gst.py")
PYTHON_INTERPRETER = "/usr/bin/python" # Ensure this interpreter matches the Speech node's environment

class ONNXCameraMaster:
    def __init__(self):
        rospy.init_node('onnx_master_controller', anonymous=True)

        self.tts_executed = False
        self.bridge = CvBridge()
        self.conf_threshold = 0.45
        self.iou_threshold = 0.5

        # Subprocess: Speech Interaction Node
        self.speech_process = None
        self.is_shutting_down_speech = False # Added: flag indicating whether speech process is shutting down
        
        # Load Params
        self.model_path = rospy.get_param('~model_path', './models/best1226.onnx')
        self.label_path = rospy.get_param('~label_path', './labels.txt')
        self.required_consecutive = rospy.get_param('~required_consecutive', 20)
        self.no_target_timeout = float(rospy.get_param('~no_target_timeout', 5.0))

        # ONNX Session
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]

        # Labels
        self.labels = []
        try:
            with open(self.label_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        except:
            rospy.logwarn("Labels file failed to load.")

        # Topics
        self.image_sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.image_callback, queue_size=1)
        self.result_pub = rospy.Publisher("/onnx_recognition/result_image", Image, queue_size=1)
        self.stop_pub = rospy.Publisher("/speech/stop", Bool, queue_size=10)

        self.detection_counter = {}
        self.last_detection_time = rospy.Time.now()

        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self.monitor_speech_output, daemon=True)
        self.monitor_thread.start()

        rospy.loginfo("ONNX Master Node Initialized.")

    def launch_speech_process(self):
        """Start the speech interaction subprocess"""
        # If shutting down, do not restart until fully exited
        if self.is_shutting_down_speech:
            # Check if the process has fully exited
            if self.speech_process and self.speech_process.poll() is not None:
                 self.is_shutting_down_speech = False
                 self.speech_process = None
            else:
                 rospy.logwarn("Waiting for previous speech process to exit fully...")
                 return False

        if self.speech_process is None or self.speech_process.poll() is not None:
            rospy.loginfo(f"Launching Speech Node: {SPEECH_SCRIPT_PATH}")
            try:
                self.speech_process = subprocess.Popen(
                    [PYTHON_INTERPRETER, "-u", SPEECH_SCRIPT_PATH],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                self.is_shutting_down_speech = False
                return True
            except Exception as e:
                rospy.logerr(f"Failed to launch speech process: {e}")
                return False
        return True

    def send_expression_to_speech(self, expression):
        """Send expression to the speech process via stdin"""
        if self.launch_speech_process():
            try:
                rospy.loginfo(f"Sending trigger to Speech Node: {expression}")
                self.speech_process.stdin.write(expression + "\n")
                self.speech_process.stdin.flush()
            except Exception as e:
                rospy.logerr(f"Failed to write to speech process: {e}")

    def graceful_stop_speech(self):
        """Send stop command to the Speech process to exit after finishing audio"""
        if self.speech_process and self.speech_process.poll() is None and not self.is_shutting_down_speech:
            rospy.loginfo("Sending Graceful SHUTDOWN command to Speech Node...")
            try:
                self.speech_process.stdin.write("CMD:SHUTDOWN\n")
                self.speech_process.stdin.flush()
                self.is_shutting_down_speech = True
            except Exception as e:
                rospy.logerr(f"Error sending shutdown: {e}")
                # If sending fails, force terminate
                self.speech_process.terminate()
                self.speech_process = None
                self.is_shutting_down_speech = False

    def monitor_speech_output(self):
        """Read and display output from the Speech process"""
        while not rospy.is_shutdown():
            # Read only when the process exists and is alive
            if self.speech_process and self.speech_process.poll() is None:
                try:
                    line = self.speech_process.stdout.readline()
                    if line:
                        print(f"[SpeechNode]: {line.strip()}")
                except:
                    pass
            else:
                # When the process exits, update state
                if self.speech_process is not None:
                    # Set to None only when a non-None process is detected to have exited
                    self.speech_process = None
                    self.is_shutting_down_speech = False
                rospy.sleep(0.5)

    # ... (letterbox, preprocess_image, postprocess, scale_coords code unchanged) ...
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2; dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def preprocess_image(self, img):
        img_lb, ratio, pad = self.letterbox(img, new_shape=(self.input_h, self.input_w))
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0), ratio, pad

    def postprocess(self, output):
        pred = np.transpose(output, (0, 2, 1))[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]; confidences = confidences[mask]; class_ids = class_ids[mask]
        if len(boxes) == 0: return None, None, None
        y = np.copy(boxes)
        y[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        y[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        y[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        y[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        boxes = y
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), self.conf_threshold, self.iou_threshold)
        if len(indices) == 0: return None, None, None
        if isinstance(indices, tuple): indices = indices[0]
        final_boxes = boxes[indices]
        final_scores = confidences[indices]
        final_ids = class_ids[indices]
        if len(final_scores) > 0:
            best = np.argmax(final_scores)
            return int(final_ids[best]), float(final_scores[best]), final_boxes[best]
        return None, None, None

    def scale_coords(self, box, ratio, pad):
        x1, y1, x2, y2 = box
        dw, dh = pad
        x1 = (x1 - dw) / ratio
        x2 = (x2 - dw) / ratio
        y1 = (y1 - dh) / ratio
        y2 = (y2 - dh) / ratio
        return [int(x1), int(y1), int(x2), int(y2)]

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError: return

        input_tensor, ratio, pad = self.preprocess_image(cv_image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        cls_id, conf, box = self.postprocess(outputs[0])

        current_id = None
        if box is not None:
            self.last_detection_time = rospy.Time.now()
            current_id = cls_id
            
            x1, y1, x2, y2 = self.scale_coords(box, ratio, pad)
            h, w = cv_image.shape[:2]
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)

            label = str(cls_id)
            if 0 <= cls_id < len(self.labels): label = self.labels[cls_id]

            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{label} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Counter Logic
        for k in list(self.detection_counter.keys()):
            if k != current_id: del self.detection_counter[k]

        if current_id is not None:
            self.detection_counter[current_id] = self.detection_counter.get(current_id, 0) + 1
            
            # Trigger only when the speech process is not shutting down and TTS not yet executed
            if self.detection_counter[current_id] == self.required_consecutive and not self.tts_executed and not self.is_shutting_down_speech:
                label = self.labels[current_id] if 0 <= current_id < len(self.labels) else str(current_id)
                rospy.loginfo(f"Target Confirmed: {label}. Pulling up Speech Interaction...")
                
                self.send_expression_to_speech(label)
                self.tts_executed = True
        else:
            self.detection_counter.clear()

        # --- Timeout & Shutdown Logic ---
        if (rospy.Time.now() - self.last_detection_time).to_sec() > self.no_target_timeout:
            if self.tts_executed:
                rospy.loginfo("Target Lost (Timeout). Initiating Shutdown Sequence.")
                self.stop_pub.publish(Bool(data=True)) # Optional: notify other nodes
                self.tts_executed = False
                self.detection_counter.clear()
                
                # Invoke graceful exit
                self.graceful_stop_speech()

        self.result_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        cv2.imshow("ONNX Master", cv_image)
        cv2.waitKey(1)

    def run(self):
        rospy.spin()
        if self.speech_process:
            self.speech_process.terminate()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        ONNXCameraMaster().run()
    except rospy.ROSInterruptException: pass