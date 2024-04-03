import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
from sensor_msgs.msg import CompressedImage 

from prediction_helper import ObjectDetection
 
class ImageSubscriber(Node):
    def __init__(self):
        self.objdec = ObjectDetection() 
        super().__init__('image_subscriber')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel_microros', 10)
        self.subscription = self.create_subscription(
            Image,
            'webcam_image',
            self.listener_callback,
            10)
        self.subscription # prevent unused variable warning
        self.bridge = CvBridge()
        self.publisher_2 = self.create_publisher(CompressedImage, 'switch_image', 10)

 
    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        results = self.objdec.predict(cv_image)
        frame, _, confidences, class_ids = self.objdec.plot_bboxes(results, cv_image)
        frame, direction, cmdvel = self.objdec.center_checker(frame, self.objdec.center_coords)
        msg = Twist()
        msg.linear.x = 0.0 # Forward speed
        msg.angular.z = cmdvel # No rotation
        self.publisher_.publish(msg)
        self.get_logger().info('I heard: "%f"' % cmdvel)
        frame = cv2.resize(frame, (160, 120))
        # Compress the frame
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]  # Adjust quality as needed
        _, compressed_data = cv2.imencode('.jpg', frame, encode_param)

        # Convert compressed data to bytes and create CompressedImage message
        image_msg = CompressedImage()
        image_msg.format = 'jpeg'
        image_msg.data = compressed_data.tobytes()
        self.publisher_2.publish(image_msg)
        # cv2.imshow('Webcam Image', frame)
        # cv2.waitKey(1)
 
def main(args=None):
    rclpy.init(args=args)
 
    image_subscriber = ImageSubscriber()
 
    rclpy.spin(image_subscriber)
 
    image_subscriber.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()