
Testing

UDP:
1. low quality (lower bitrate)
    on pi:
    rpicam-vid -t 0 --inline --width 640 --height 480 --framerate 30 --bitrate 1500000 --intra 1 --low-latency -o udp://192.168.0.7:1234 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
    on pc:
    ffplay -i udp://@:1234 -fflags nobuffer+fastseek -flags low_delay -framedrop -probesize 32 -sync ext
2. better quality (higher bitrate, too high bitrate can lead to lag though)
    on pi:
    rpicam-vid -t 0 --inline --width 1280 --height 720 --framerate 30 --bitrate 5000000 --intra 15 --low-latency -o udp://192.168.0.7:1234 --post-process-file /usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json
    on pc:
    ffplay -i udp://@:1234 -fflags nobuffer+fastseek -flags low_delay -framedrop -probesize 32 -sync ext