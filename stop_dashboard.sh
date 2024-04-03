#!/bin/bash


# kill python3 server
http_server_pid=$(ps aux |grep "python3 -m http.server" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$http_server_pid" ]]
then
  kill $http_server_pid
  echo "killed http server"
fi

# kill rosbridge launch
rosbridge_server_pid=$(ps aux | grep "/usr/bin/python3 /opt/ros/humble/bin/ros2 launch rosbridge_server rosbridge_websocket_launch.xml" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$rosbridge_server_pid" ]]
then
  kill $rosbridge_server_pid
  echo "killed rosbridge launch"
fi

# kill rosbridge websocket
rosbridge_websocket_pid=$(ps aux | grep "python3 /opt/ros/humble/lib/rosbridge_server/rosbridge_websocket" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$rosbridge_websocket_pid" ]]
then
  kill -9 $rosbridge_websocket_pid # -9 to kill the parent too | without this it dosen't work
  echo "killed rosbridge websocket"
fi

# kill rosapi
ros_api_pid=$(ps aux | grep "python3 /opt/ros/humble/lib/rosapi/rosapi_node" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$ros_api_pid" ]]
then
  kill -9 $ros_api_pid # -9 to kill the parent too | without this it dosen't work
  echo "killed ros_api_pid"
fi


# kill rover_latency_node
ping_pid=$(ps aux |grep "python3 scripts/rover_latency_node" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$ping_pid" ]]
then
  kill $ping_pid
  echo "killed rover_latency_node"
fi


# kill speak_node
speak_pid=$(ps aux |grep "python3 scripts/speak_node" |awk '{for(i=1;i<=NF;i++) if ($i ~ /^[0-9]+$/) print $i}' |head -n1)
if [[ -n "$speak_pid" ]]
then
  kill $speak_pid
  echo "killed speak_node"
fi