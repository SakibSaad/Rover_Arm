#!/bin/bash


# get gps_data.json
scripts/get_coordinate_file.sh

# server setup
ip_var=$(hostname -I | awk '{print $1}')
rm -rf nohup.out
nohup ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
nohup python3 -m http.server &
nohup python3 scripts/rover_latency_node.py &
nohup python3 scripts/speak_node.py &

echo "




	Dashboard URL: $ip_var:8000




"
