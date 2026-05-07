import rclpy
from rclpy.node import Node


class StateMachineNode(Node):
    def __init__(self):
        super().__init__("state_machine_node")
        self.state = "IDLE"
        self.get_logger().info(f"State machine started in {self.state}")


def main(args=None):
    rclpy.init(args=args)
    node = StateMachineNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
