#!/bin/bash

# server setup
ip_var=$(hostname -I | awk '{print $1}')
rm -rf nohup.out
nohup ros2 launch rosbridge_server rosbridge_websocket_launch.xml &
nohup python3 -m http.server &

echo "




	Dashboard URL: $ip_var:8000




"
