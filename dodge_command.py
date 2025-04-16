import glob
import sys

import serial
import json
import time


def list_serial_ports():
    """Lists available serial ports on multiple platforms"""
    if sys.platform.startswith('win'):
        # Windows
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux'):
        # Linux
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        # macOS
        ports = glob.glob('/dev/tty.*')
        ports.extend(glob.glob('/dev/cu.*'))
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except(OSError, serial.SerialException):
            pass
    return result


class DodgeCommandModule:
    """
    Module for sending dodge commands to the robot based on the trajectory predictions.
    """
    def __init__(self, port=None, baud_rate=115200, robot_width=15):
        """Initialize the dodge command module"""
        # Serial communication setup
        self.port = port
        self.baud_rate = baud_rate
        self.ser = None
        self.connected = False

        # Robot parameters
        self.robot_width = robot_width
        self.robot_position = None
        self.target_position = None

        # Dodge state tracking
        self.is_dodging = False
        self.dodge_start_time = None
        self.last_command_time = 0
        self.command_cooldown = 0.3 # Minimum seconds between commands
        self.dodge_timeout = 3.0 # Maximum dodge duration in seconds

        # Auto select port if not specified
        if port is None:
            self.port = self.select_serial_port()

        # Establish serial connection
        if self.port:
            self.connect()

    def select_serial_port(self):
        available_ports = list_serial_ports()

        if not available_ports:
            print('No serial ports found. Check connections')
            return None

        print('Available serial ports: ')
        for i, port in enumerate(available_ports):
            print(f"{i + 1}. {port}")

        try:
            selection = int(input('Select port number (or 0 to cancel): '))
            if selection == 0:
                return None
            if 1 <= selection <= len(available_ports):
                return available_ports[selection - 1]
            else:
                print('Invalid selection')
                return None
        except ValueError:
            print('Invalid input.')
            return None

    def connect(self):
        """Establish serial connection with Arduino"""
        if not self.port:
            print('No port specified. Cannot connect.')
            return False

        try:
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)
            self.connected = True
            print(f"Connected to Arduino on {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False

    def send_command(self, command_dict):
        """Send a command to Arduino"""
        if not self.connected:
            print("Not connected to Arduino")
            return False

        try:
            # Convert to JSON and add newline as terminator
            command_json = json.dumps(command_dict) + "\n"
            self.ser.write(command_json.encode())
            self.last_command_time = time.time()
            print(f"Sent: {command_json.strip()}")

            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False

    def update_robot_position(self, position_cm):
        """Update known position of the robot from vision system and check dodge status"""
        previous_position = self.robot_position
        self.robot_position = position_cm

        if self.is_dodging:
            if self.target_position is not None:
                if abs(position_cm - self.target_position) < 2.0: # Within 2 cm tolerance
                    print(f"Dodge complete - Robot reached target position: {position_cm:.1f} cm")
                    self.is_dodging = False
                    return

    def process_collision(self, landing_point, time_to_impact=None):
        """
                Process potential collision and decide whether to dodge.

                Args:
                    landing_point: (x, y) coordinates of predicted landing point
                    time_to_impact: Optional time to impact in seconds

                Returns:
                    bool: True if dodge command was sent, False otherwise
        """
        # Skip if we're already dodging or don't have necessary data
        if self.is_dodging or landing_point is None or self.robot_position is None:
            return False

        # Throttle commands
        current_time = time.time()
        if current_time - self.last_command_time < self.command_cooldown:
            return False

        landing_x  = landing_point[0]

        # Calculate distance from landing point to robot center
        distance = abs(landing_x - self.robot_position)

        collision_threshold =  (self.robot_width / 2) + 3

        if distance >= collision_threshold:
            # No collision predicted
            return False

        dodge_distance = collision_threshold - distance + 5

        dodge_direction = "right" if landing_x < self.robot_position else "left"

        # Calculate urgency based on time to impact
        urgency = 0.8 # default high urgency
        urgency = min(1.0, max(0.6, 1.0 / time_to_impact + 0.5))

        command = {
            "command": "dodge",
            "parameters": {
                "direction": dodge_direction,
                "distance": float(dodge_distance),
                "urgency": float(urgency)
            }
        }

        # Add time to impact if available
        if time_to_impact is not None:
            command["parameters"]["time_to_impact"] = float(time_to_impact)

        # Send command
        success = self.send_command(command)

        if success:
            self.is_dodging = True
            self.dodge_start_time = time.time()

            # Calculate target position
            dir_multiplier = 1 if dodge_direction == "right" else -1
            self.target_position = self.robot_position + (dir_multiplier * dodge_distance)

            print(f"Dodge initiated: {dodge_direction} by {dodge_distance:.1f}cm, "
                  f"urgency: {urgency:.2f}, target: {self.target_position:.1f}cm")
            return True

    def test_dodge_left(self, distance=20.0):
        """Send a test dodge command to the left."""
        success = self.send_command({
            "command": "dodge",
            "parameters": {
                "direction": "left",
                "distance": float(distance)
            }
        })

        if success:
            self.is_dodging = True
            self.dodge_start_time = time.time()
            self.target_position = self.robot_position - distance

        return success

    def test_dodge_right(self, distance=20.0):
        """Send a test dodge command to the right."""
        success = self.send_command({
            "command": "dodge",
            "parameters": {
                "direction": "right",
                "distance": float(distance)
            }
        })

        if success:
            self.is_dodging = True
            self.dodge_start_time = time.time()
            self.target_position = self.robot_position + distance

        return success

    def emergency_stop(self):
        """Send emergency stop command."""
        self.is_dodging = False
        return self.send_command({"command": "stop", "parameters": {"emergency": True}})

    def close(self):
        """Close the serial connection."""
        if self.connected and self.ser:
            self.ser.close()
            self.connected = False
            print("Serial connection closed")
