# Bundled custom message definitions

The six `px4_msgs/msg/*.msg` files were copied from the recording workspace:

`Seans_Project/DASC/3DGS simulator_Sim2Real/quad_px4_ros2_jumpstart/colcon_ws/src/px4_msgs/msg/`

The local Git revision reported at that source path was `36f57c7e61c0adb997015f7148696dd89911f616` when copied. This identifies the local workspace revision, not necessarily an upstream PX4 release. The accompanying PX4 license is preserved in `px4_msgs/`.

These definitions decode the included October 13 recording. Do not assume they match a future PX4 firmware/message build. Use `--msg-dir ROOT` to supply matching `ROOT/px4_msgs/msg/*.msg` definitions when needed. Standard messages come from the Rosbags ROS 2 Humble type store, or definitions embedded in the recording.
