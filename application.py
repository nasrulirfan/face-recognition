
import zlib
from werkzeug.utils import secure_filename
from flask import Response
import cv2
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from tempfile import mkdtemp
from werkzeug.exceptions import default_exceptions, HTTPException, InternalServerError
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
import face_recognition
from PIL import Image
from base64 import b64encode, b64decode
import re
from helpers import apology, login_required
from flask import Flask, render_template, request, url_for, redirect, session
import pymongo
import bcrypt
import os

# Configure application
app = Flask(__name__)
app.secret_key = "super secret"
# configure flask-socketio

# Ensure templates are auto-reloaded
app.config["TEMPLATES_AUTO_RELOAD"] = True


# client = pymongo.MongoClient("mongodb://mongo:4LL2HELS76Gg6Lp2mHQH@containers-us-west-102.railway.app:6671")
client = pymongo.MongoClient("mongodb://localhost:27017/mydatabase")
db = client.get_database('login')
users = db.users

# Ensure responses aren't cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response


# Custom filter


# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"

@app.route("/")
@login_required
def home():
    return redirect("/home")


@app.route("/home")
@login_required
def index():
    return render_template("index.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    """Log user in"""

    # Forget any user_id
    session.clear()

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")

        # Ensure username was submitted
        if not input_username:
            return render_template("login.html", messager=1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("login.html", messager=2)

        # Query database for username
        name_found = users.find_one({"name": input_username})
        if name_found:
            name_val = name_found['name']
            passwordcheck = name_found['password']

            if bcrypt.checkpw(input_password.encode('utf-8'), passwordcheck):
                # Remember which user has logged in
                session["user_id"] = name_val
                # Redirect user to home page
                return redirect("/")

            else:
                return render_template("login.html", messager=3)
        else:
            return render_template("login.html", messager=4)


    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("login.html")

@app.route("/success")
def success():

    return render_template("success.html")
@app.route("/logout")
def logout():
    """Log user out"""

    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return redirect("/")


@app.route("/register", methods=["GET", "POST"])
def register():
    """Register user"""

    # User reached route via POST (as by submitting a form via POST)
    if request.method == "POST":

        # Assign inputs to variables
        input_username = request.form.get("username")
        input_password = request.form.get("password")
        input_confirmation = request.form.get("confirmation")

        # Ensure username was submitted
        if not input_username:
            return render_template("register.html", messager=1)

        # Ensure password was submitted
        elif not input_password:
            return render_template("register.html", messager=2)

        # Ensure passwsord confirmation was submitted
        elif not input_confirmation:
            return render_template("register.html", messager=4)

        elif not input_password == input_confirmation:
            return render_template("register.html", messager=3)

        # Query database for username
        user_found = users.find_one({"name": input_username})
        if user_found:
            return render_template("register.html", messager=5)

        # Ensure username is not already taken

        # Query database to insert new user
        else:
            hashed = bcrypt.hashpw(input_password.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': input_username,'password': hashed}
            users.insert_one(user_input)

            new_user = users.find_one({'name': input_username})
            if new_user:
                # Keep newly registered user logged in
                session["user_id"] = new_user["name"]

            # Flash info for the user
            flash(f"Registered as {input_username}")

            # Redirect user to homepage
            return redirect("/")

    # User reached route via GET (as by clicking a link or via redirect)
    else:
        return render_template("register.html")


@app.route("/facelogin", methods=["GET", "POST"])
def facelogin():
    session.clear()
    if request.method == "POST":
        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        username = request.form.get("name")
        id_ = username
        compressed_data = zlib.compress(encoded_image, 5)
        uncompressed_data = zlib.decompress(compressed_data)

        decoded_data = b64decode(uncompressed_data)

        #Check whether user has set face recognition yet
        user_found = users.find_one({"name": id_})
        trained = user_found["trained"]

        if (trained):
            new_image_handle = open('./static/face/captured-' + str(id_) + '.jpg', 'wb')
            new_image_handle.write(decoded_data)
            new_image_handle.close()
            predict = face_recognition.predict_faces(
            './static/face/captured-' + str(id_) + '.jpg')
            os.remove('./static/face/captured-' + str(id_) + '.jpg')
            print(predict)
            if (predict == id_):
                session["user_id"] = user_found["name"]
                return redirect("/success")
            else:
                return render_template("camera.html", message=3)
        else:
            return render_template("camera.html", message=5)
    else:
        return render_template("camera.html")


@app.route("/facesetup", methods=["GET", "POST"])
def facesetup():
    if request.method == "POST":

        encoded_image = (request.form.get("pic") + "==").encode('utf-8')
        user_name=session["user_id"]
        user_found = users.find_one({"name":user_name })
        id_ = user_found["name"]

        # Create a folder for the user face
        path = os.path.join('./static/face', id_)
        os.makedirs(path, exist_ok=True)

        compressed_data = zlib.compress(encoded_image, 5)
        uncompressed_data = zlib.decompress(compressed_data)
        decoded_data = b64decode(uncompressed_data)

        #Close temporarily add some images to the faces
        new_image_handle = open('./static/face/' + str(id_) + '/' + str(id_) + '.jpg', 'wb')
        new_image_handle.write(decoded_data)
        new_image_handle.close()

        ##Preprocess stored image
        if face_recognition.preprocess_faces():
            ##Train Custom Image
            if face_recognition.train_save_model():
                users.update_one({ "name": id_ }, { "$set": { "trained": True }})
                render_template("camera.html", message=7)
                return redirect("/home")
            else:
                return render_template("camera.html", message=8)

        ### PAST
        # else: 
        #     return render_template("camera.html", message=9)

        # image_of_user = face_recognition.load_image_file(
        #     './static/face/' + str(id_) + '.jpg')
        
        # listFaces = os.listdir('./static/face')
        # user_face_encoding = face_recognition.face_encodings(image_of_user)[0]
        # print(listFaces)
        # flag = False
        # for face in listFaces:
        #     toCheck = face_recognition.load_image_file('./static/face/' + face)
            
        #     check_image = cv2.cvtColor(toCheck, cv2.COLOR_BGR2RGB)
        #     check_image_encoding = face_recognition.face_encodings(check_image)[0]
        #     #  compare faces
        #     results = face_recognition.compare_faces(
        #         [user_face_encoding], check_image_encoding)

        #     if results[0] == True:
        #         print('results is true')
        #         flag = True
        #         break

        # if flag:
        #     print("inside flag")
        #     os.remove('./static/face/' + str(id_) + '.jpg')
        #     # users.delete_one({"name": user_name})
        #     return render_template("camera.html", message=6)

        # return redirect("/home")

    else:
        return render_template("face.html")


def errorhandler(e):
    """Handle error"""
    if not isinstance(e, HTTPException):
        e = InternalServerError()
    return render_template("error.html", e=e)


# Listen for errors
for code in default_exceptions:
    print(code)
    app.errorhandler(code)(errorhandler)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=9874)






