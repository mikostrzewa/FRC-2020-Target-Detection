#!/bin/bash

v4l2-ctl --set-ctrl=gain=00
v4l2-ctl --set-ctrl=exposure_auto=1
v4l2-ctl --set-ctrl=exposure_absolute=20
v4l2-ctl --set-ctrl=zoom_absolute=130
v4l2-ctl --set-ctrl=contrast=255
v4l2-ctl --set-ctrl=saturation=255

python3 TargetDetection.py
