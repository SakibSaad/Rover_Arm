import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
 
class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.publisher_ = self.create_publisher(Image, 'webcam_image', 10)
        self.bridge = CvBridge()
        self.timer_period = 0.05 # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

    
    
 
    def timer_callback(self):
        ret, frame = cv2.VideoCapture(2).read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        else:
            self.get_logger().error("Failed to capture frame")
 
def main(args=None):
    rclpy.init(args=args)
 
    webcam_publisher = WebcamPublisher()
 
    rclpy.spin(webcam_publisher)
 
    webcam_publisher.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()

     

