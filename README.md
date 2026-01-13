to run this repo,

orangepi@orangepi5plus:~/dx-all-suite/dx-runtime/dx_app$ ./bin/run_detector -c ./bin/run_detector -c /home/orangepi/Desktop/umraiz/dx_Jan6_yolov8.json

This also works
I need to activate this 
(dx-venv)
source /home/orangepi/dx-all-suite/dx-venv/bin/activate
python3 -Xfaulthandler /home/orangepi/dx-all-suite/dx-runtime/dx_app/templates/python/yolov8m.py --config /home/orangepi/dx-all-suite/dx-runtime/dx_app/example/run_detector/yolov5s3_example.json --output_dir /home/orangepi/Desktop/umraiz/output_images_yolov5/


