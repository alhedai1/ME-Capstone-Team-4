import rclpy
from rclpy.node import Node


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__("motor_driver_node")
        self.get_logger().info("Motor driver placeholder started with motors disabled")


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
