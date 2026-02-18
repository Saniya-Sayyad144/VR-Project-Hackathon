Command to run the project in the terminal:
.\.venv\Scripts\Activate
pip install flask flask-socketio opencv-python mediapipe numpy

If you run into error of: AttributeError: module 'mediapipe' has no attribute 'solutions'

Run the following command:
pip uninstall -y mediapipe protobuf
pip install mediapipe==0.10.9 protobuf==3.20.3

Then final command:
python app.py

After that it will give a http://10.***.***.***:5000
Copy that IP address into chrome and run.

Also allow AR/VR, In the chrome go to: chrome://flags

Search for Insecure origins treated as secure and add your IP Address. Click on Enabled and then on relaunch.
Again run the http://10.***.***.***:5000. your project will run.
