import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import time
from ultralytics import YOLO
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf2_ros
import os
import shutil

class YOLOv8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')
        self.subscription = self.create_subscription(
            Image,
            'camera1/image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        
        try:
            self.model = YOLO('office_robot_pkg/best.pt')  
        except Exception as e:
            self.get_logger().error(f"Error loading model: {e}")
            raise
        
        self.publisher_ = self.create_publisher(String, 'objects_detected', 10)
        self.last_detection_time = 0
        self.viewer_position = None
        self.detection_interval = 10 
        self.confidence_threshold = 0.8
        self.position_threshold = 0.5 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.clear_objects_file()

    def clear_objects_file(self):
        try:
            with open('send_server/objects.txt', 'w') as f:
                f.truncate(0)
            self.get_logger().info("Cleared contents of 'send_server/objects.txt'")

            detected_objects_dir = 'send_server/detected_objects'
            if os.path.exists(detected_objects_dir):
                shutil.rmtree(detected_objects_dir)
            os.makedirs(detected_objects_dir)
            self.get_logger().info(f"Cleared contents of '{detected_objects_dir}'")
        except Exception as e:
            self.get_logger().error(f"Error clearing files: {e}")

    def listener_callback(self, msg):
        self.update_viewer_position()
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_interval:
            return

        self.last_detection_time = current_time
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        try:
            results = self.model(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error during detection: {e}")
            return

        self.save_results_to_file(cv_image, results)

    def update_viewer_position(self):
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform('odom', 'base_link', now)
            self.viewer_position = np.array([trans.transform.translation.x,
                                             trans.transform.translation.y,
                                             trans.transform.translation.z])
        except Exception as e:
            self.get_logger().info(f'Failed to update viewer position: {str(e)}')

    def save_results_to_file(self, image, results):
        detected_names = set()
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy
                names = result.names
                confidences = result.boxes.conf

                for i in range(len(boxes)):
                    confidence = confidences[i].cpu().numpy()
                    if confidence < self.confidence_threshold:
                        continue

                    box = boxes[i].cpu().numpy()
                    name = names[int(result.boxes.cls[i])]
                    detected_names.add(name)

                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(image, f"{name} {confidence:.2f}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if detected_names:
            self.publisher_.publish(String(data=", ".join(detected_names)))
            
            detected_objects_dir = 'send_server/detected_objects'
            image_files = os.listdir(detected_objects_dir)
            image_count = len(image_files)
            image_filename = os.path.join(detected_objects_dir, f'detected_objects_{image_count + 1}.jpg')
            
            try:
                cv2.imwrite(image_filename, image)
                self.get_logger().info(f"Saved image with detected objects to '{image_filename}'")
            except Exception as e:
                self.get_logger().error(f"Error saving image: {e}")

            if self.should_save_position():
                try:
                    with open('send_server/objects.txt', 'a') as f:
                        f.write("Detected Objects:\n")
                        for name in detected_names:
                            f.write(f"{name}\n")
                        f.write("\nViewer Position:\n")
                        if self.viewer_position is not None:
                            f.write(f"X: {self.viewer_position[0]}, Y: {self.viewer_position[1]}, Z: {self.viewer_position[2]}\n")
                        else:
                            f.write("Viewer position not available\n")
                        f.write("\n")
                    self.get_logger().info("Appended detected object names and viewer position to 'send_server/objects.txt'")
                except Exception as e:
                    self.get_logger().error(f"Error appending detected object names and viewer position: {e}")

    def should_save_position(self):
        if not os.path.exists('send_server/objects.txt'):
            return True

        try:
            with open('send_server/objects.txt', 'r') as f:
                lines = f.readlines()
                positions = []
                for line in lines:
                    if line.startswith("X: "):
                        pos = list(map(float, line.strip().replace("X: ", "").replace("Y: ", "").replace("Z: ", "").split(",")))
                        positions.append(np.array(pos))

            if not positions:
                return True

            for pos in positions:
                if np.linalg.norm(self.viewer_position - pos) < self.position_threshold:
                    return False
            return True
        except Exception as e:
            self.get_logger().error(f"Error reading viewer positions from file: {e}")
            return True

def main(args=None):
    rclpy.init(args=args)
    yolov8_detector = YOLOv8Detector()
    rclpy.spin(yolov8_detector)
    yolov8_detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
