import base64
import os

from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np

app = Flask(__name__)
UPLOAD_PATH = "uploads/temp.png"
ALLOWED_EXT = {"jpg", "jpeg", "png", "tiff"}

def allowed(filename):
    return filename.split(".")[-1].lower() in ALLOWED_EXT


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="Файл не найден"), 400

    file = request.files["image"]

    if not allowed(file.filename):
        return jsonify(error="Неподдерживаемый формат"), 400

    try:
        image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)
        cv2.imwrite(UPLOAD_PATH, img)
        return jsonify(message="Изображение загружено")
    except:
        return jsonify(error="Ошибка при загрузке файла"), 400


@app.route("/resize", methods=["POST"])
def resize():
    data = request.json
    fx, fy = data["fx"], data["fy"]

    img = cv2.imread(UPLOAD_PATH)
    interp = cv2.INTER_CUBIC if fx > 1 or fy > 1 else cv2.INTER_LINEAR

    resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)
    cv2.imwrite(UPLOAD_PATH, resized)

    return jsonify(message="Размер изменён")


# 🔹 Обрезка
@app.route("/crop", methods=["POST"])
def crop():
    data = request.json
    x, y, w, h = data.values()

    img = cv2.imread(UPLOAD_PATH)
    H, W = img.shape[:2]

    if x < 0 or y < 0 or x+w > W or y+h > H:
        return jsonify(error="Координаты вне изображения"), 400

    cropped = img[y:y+h, x:x+w]
    cv2.imwrite(UPLOAD_PATH, cropped)

    return jsonify(message="Фрагмент вырезан")


@app.route("/flip", methods=["POST"])
def flip():
    mode = request.json["mode"]  # 0,1,-1
    img = cv2.imread(UPLOAD_PATH)
    flipped = cv2.flip(img, mode)
    cv2.imwrite(UPLOAD_PATH, flipped)
    return jsonify(message="Отражение выполнено")

@app.route("/rotate", methods=["POST"])
def rotate():
    angle = request.json["angle"]
    img = cv2.imread(UPLOAD_PATH)
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    cv2.imwrite(UPLOAD_PATH, rotated)
    return jsonify(message="Поворот выполнен")


@app.route("/brightness", methods=["POST"])
def brightness():
    alpha = request.json["contrast"]
    beta = request.json["brightness"]

    img = cv2.imread(UPLOAD_PATH)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    cv2.imwrite(UPLOAD_PATH, adjusted)
    return jsonify(message="Изменение применено")


@app.route("/color", methods=["POST"])
def color():
    r, g, b = request.json.values()
    img = cv2.imread(UPLOAD_PATH)

    img[:,:,0] = np.clip(img[:,:,0] + b, 0, 255)
    img[:,:,1] = np.clip(img[:,:,1] + g, 0, 255)
    img[:,:,2] = np.clip(img[:,:,2] + r, 0, 255)

    cv2.imwrite(UPLOAD_PATH, img)
    return jsonify(message="Цветовой баланс изменён")


@app.route("/noise", methods=["POST"])
def noise():
    img = cv2.imread(UPLOAD_PATH)
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    noisy = cv2.add(img, noise)
    cv2.imwrite(UPLOAD_PATH, noisy)
    return jsonify(message="Шум добавлен")


@app.route("/blur", methods=["POST"])
def blur():
    k = request.json["kernel"]
    img = cv2.imread(UPLOAD_PATH)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    cv2.imwrite(UPLOAD_PATH, blurred)
    return jsonify(message="Размытие применено")


@app.route("/image")
def image():
    return send_file(UPLOAD_PATH, mimetype="image/png")

@app.route("/mask_crop", methods=["POST"])
def mask_crop():
    data = request.json["mask"]

    # убрать префикс base64
    mask_data = base64.b64decode(data.split(",")[1])
    mask_np = np.frombuffer(mask_data, np.uint8)
    mask = cv2.imdecode(mask_np, cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(UPLOAD_PATH)

    if mask.shape != img.shape[:2]:
        return jsonify(error="Размер маски не совпадает с изображением"), 400

    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(UPLOAD_PATH, result)

    return jsonify(message="Произвольная область вырезана")

@app.route("/save", methods=["POST"])
def save():
    data = request.json
    format = data.get("format", "png").lower()
    quality = int(data.get("quality", 95))

    img = cv2.imread(UPLOAD_PATH)

    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    filename = f"result.{format}"
    save_path = os.path.join("uploads", filename)

    try:
        if format == "jpg" or format == "jpeg":
            cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif format == "png":
            cv2.imwrite(save_path, img)
        elif format == "tiff":
            cv2.imwrite(save_path, img)
        else:
            return jsonify(error="Неподдерживаемый формат"), 400

        return send_file(save_path, as_attachment=True)

    except:
        return jsonify(error="Ошибка при сохранении файла"), 500




if __name__ == "__main__":
    app.run(debug=True)
