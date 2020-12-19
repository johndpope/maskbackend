from flask import Flask, render_template, request, send_file

from getnewimage import getnewImage
import base64
from io import BytesIO
import dnnlib
from PIL import Image
import os
import gdown
import subprocess


url = "https://drive.google.com/uc?id=1S1INWfG8G3sUSFqr4MKW2FLAwhu-pvF_"
output = 'FFHQ-CartoonsAlignedHQ36v2.pkl'
if os.path.isfile(output) is False:
    print('file not found, downloading from google drive')
    gdown.download(url, output)



# initialize tensorflow session

dnnlib.tflib.init_tf()


# initialize app

app = Flask(__name__)


# convert returned image
def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=90)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

# serve index page
@app.route('/')
def dynamic_page():
    return render_template('index.html')

# serve image
@app.route('/generate', methods=['POST', 'GET'])
def getimage():
    data = getnewImage()
    return serve_pil_image(data)

if __name__ == '__main__':
    # subprocess.run('kill $(lsof -t -i :8000)', shell=True)
    app.run(host='0.0.0.0', port='8080', debug=True, threaded=False)