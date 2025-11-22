import os
from flask import Flask, render_template, request
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def set_folder():
    files = []
    folder_path = ''
    if request.method == 'POST':
        folder_path = request.form['folder_path']
        try:
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        except Exception as e:
            files = []
    return render_template('index.html', files=files, folder_path=folder_path)

@app.route('/predict', methods=['POST'])
def predict():
    folder_path = request.form['folder_path']
    filename = request.form['filename']
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as f:
        img_bytes = f.read()
    yolo_server_url = 'http://localhost:5000/detect'  # 실제 IP와 PORT로 변경
    res = requests.post(yolo_server_url, data=img_bytes)
    # result = res.json()
    # return render_template('result.html', result=result, filename=filename, folder_path=folder_path)
    return render_template('result.html', result=res.text, filename=filename, folder_path=folder_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
