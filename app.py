import os

from flask import Flask, request, jsonify, send_file, render_template
import cv2
import numpy as np


app = Flask(__name__)
UPLOAD_PATH = "uploads/temp.png"
HISTORY_DIR = "uploads/history"
ORIGINAL_PATH = "uploads/original.png"

os.makedirs(HISTORY_DIR, exist_ok=True)
ALLOWED_EXT = {"jpg", "jpeg", "png", "tiff"}

def allowed(filename):
    return filename.split(".")[-1].lower() in ALLOWED_EXT

def save_state():
    count = len(os.listdir(HISTORY_DIR))
    cv2.imwrite(f"{HISTORY_DIR}/{count}.png", cv2.imread(UPLOAD_PATH))



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="Файл не найден"), 400

    file = request.files["image"]

    try:
        image = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(image, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify(error="Ошибка чтения изображения"), 400

        cv2.imwrite(UPLOAD_PATH, img)
        cv2.imwrite(ORIGINAL_PATH, img)

        for f in os.listdir(HISTORY_DIR):
            os.remove(os.path.join(HISTORY_DIR, f))

        return jsonify(message="Изображение загружено")

    except:
        return jsonify(error="Ошибка загрузки"), 400



@app.route("/resize", methods=["POST"])
def resize():
    save_state()
    data = request.json
    fx, fy = data["fx"], data["fy"]

    img = cv2.imread(UPLOAD_PATH)
    interp = cv2.INTER_CUBIC if fx > 1 or fy > 1 else cv2.INTER_LINEAR

    resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)
    cv2.imwrite(UPLOAD_PATH, resized)

    return jsonify(message="Размер изменён")


@app.route("/crop", methods=["POST"])
def crop():
    save_state()
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
    save_state()
    mode = request.json["mode"]  # 0,1,-1
    img = cv2.imread(UPLOAD_PATH)
    flipped = cv2.flip(img, mode)
    cv2.imwrite(UPLOAD_PATH, flipped)
    return jsonify(message="Отражение выполнено")

@app.route("/rotate", methods=["POST"])
def rotate():
    save_state()
    angle = request.json["angle"]
    img = cv2.imread(UPLOAD_PATH)
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

    cv2.imwrite(UPLOAD_PATH, rotated)
    return jsonify(message="Поворот выполнен")


@app.route("/brightness", methods=["POST"])
def brightness():
    save_state()
    alpha = request.json["contrast"]
    beta = request.json["brightness"]

    img = cv2.imread(UPLOAD_PATH)
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    cv2.imwrite(UPLOAD_PATH, adjusted)
    return jsonify(message="Изменение применено")


@app.route("/color", methods=["POST"])
def color():
    save_state()
    r, g, b = request.json.values()
    img = cv2.imread(UPLOAD_PATH)

    img[:, :, 0] = cv2.add(img[:, :, 0], b)
    img[:, :, 1] = cv2.add(img[:, :, 1], g)
    img[:, :, 2] = cv2.add(img[:, :, 2], r)

    cv2.imwrite(UPLOAD_PATH, img)
    return jsonify(message="Цветовой баланс изменён")


@app.route("/noise", methods=["POST"])
def noise():
    save_state()
    data = request.json
    noise_type = data.get("type", "gaussian")
    amount = float(data.get("amount", 0.02))  # интенсивность

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    if noise_type == "gaussian":
        mean = 0
        sigma = amount * 255
        gauss = np.random.normal(mean, sigma, img.shape).astype(np.int16)
        noisy = img.astype(np.int16) + gauss
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    elif noise_type == "sp":
        noisy = img.copy()
        h, w, _ = img.shape
        num = int(amount * h * w)

        coords = (
            np.random.randint(0, h, num),
            np.random.randint(0, w, num)
        )
        noisy[coords] = 255

        coords = (
            np.random.randint(0, h, num),
            np.random.randint(0, w, num)
        )
        noisy[coords] = 0

    else:
        return jsonify(error="Неизвестный тип шума"), 400

    cv2.imwrite(UPLOAD_PATH, noisy)
    return jsonify(message="Шум добавлен")


@app.route("/blur", methods=["POST"])
def blur():
    save_state()
    data = request.json
    k = int(data.get("kernel", 5))
    blur_type = data.get("type", "gaussian")
    print(k, blur_type)

    if k % 2 == 0 or k <= 1:
        return jsonify(error="Размер ядра должен быть нечётным и > 1"), 400

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    if blur_type == "average":
        result = cv2.blur(img, (k, k))

    elif blur_type == "gaussian":
        result = cv2.GaussianBlur(img, (k, k), 0)

    elif blur_type == "median":
        result = cv2.medianBlur(img, k)

    else:
        return jsonify(error="Неизвестный тип фильтра"), 400

    cv2.imwrite(UPLOAD_PATH, result)
    return jsonify(message="Размытие применено")


@app.route("/image")
def image():
    return send_file(UPLOAD_PATH, mimetype="image/png")

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

@app.route("/undo", methods=["POST"])
def undo():
    files = sorted(os.listdir(HISTORY_DIR), key=lambda x: int(x.split(".")[0]))

    if not files:
        return jsonify(error="Нет действий для отмены"), 400

    last = files[-1]
    last_path = os.path.join(HISTORY_DIR, last)

    img = cv2.imread(last_path)
    cv2.imwrite(UPLOAD_PATH, img)

    os.remove(last_path)
    return jsonify(message="Последнее действие отменено")

@app.route("/reset", methods=["POST"])
def reset():
    if not os.path.exists(ORIGINAL_PATH):
        return jsonify(error="Оригинал не найден"), 400

    img = cv2.imread(ORIGINAL_PATH)
    cv2.imwrite(UPLOAD_PATH, img)

    # очистить историю
    for f in os.listdir(HISTORY_DIR):
        os.remove(os.path.join(HISTORY_DIR, f))

    return jsonify(message="Изображение восстановлено")

@app.route("/colorspace", methods=["POST"])
def change_colorspace():
    data = request.json
    space = data.get("space")

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    save_state()

    if space == "hsv":
        result = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    elif space == "bgr":
        result = cv2.cvtColor(img, cv2.COLOR_HLS2BGR)

    else:
        return jsonify(error="Неизвестное цветовое пространство"), 400

    cv2.imwrite(UPLOAD_PATH, result)
    return jsonify(message="Цветовое пространство изменено")

@app.route("/find_object", methods=["POST"])
def find_object():
    data = request.json
    color = data.get("color")        # [R, G, B] или [H, S, V]
    space = data.get("space", "rgb")
    mode = data.get("mode", "box")   # box | crop
    tol = int(data.get("tolerance", 20))

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    save_state()

    if space == "hsv":
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else:
        img_cs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    lower = np.array([max(0, c - tol) for c in color])
    upper = np.array([min(255, c + tol) for c in color])

    mask = cv2.inRange(img_cs, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return jsonify(error="Объект не найден"), 400

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    if mode == "box":
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        result = img

    elif mode == "crop":
        result = img[y:y+h, x:x+w]

    else:
        return jsonify(error="Неизвестный режим"), 400

    cv2.imwrite(UPLOAD_PATH, result)

    return jsonify(
        message="Объект найден",
        x=x, y=y, width=w, height=h
    )

@app.route("/edges", methods=["POST"])
def edges():
    data = request.json
    method = data.get("method", "sobel")

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    save_state()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if method == "sobel":
        ksize = int(data.get("ksize", 3))

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

        mag = cv2.magnitude(gx, gy)
        mag = cv2.convertScaleAbs(mag)

        result = mag

    elif method == "canny":
        t1 = int(data.get("t1", 100))
        t2 = int(data.get("t2", 200))

        result = cv2.Canny(gray, t1, t2)

    else:
        return jsonify(error="Неизвестный метод"), 400

    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(UPLOAD_PATH, result)

    return jsonify(message="Выделение границ выполнено")

@app.route("/segment", methods=["POST"])
def segment():
    data = request.json
    method = data.get("method")

    img = cv2.imread(UPLOAD_PATH)
    if img is None:
        return jsonify(error="Изображение не найдено"), 400

    save_state()
    h, w = img.shape[:2]

    # === 1. K-Means (OpenCV) ===
    if method == "kmeans":
        K = int(data.get("k", 3))
        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    10, 1.0)

        _, labels, centers = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        centers = np.uint8(centers)
        result = centers[labels.flatten()].reshape((h, w, 3))

    # === 2. Mean Shift (OpenCV) ===
    elif method == "meanshift":
        sp = int(data.get("sp", 20))
        sr = int(data.get("sr", 30))

        result = cv2.pyrMeanShiftFiltering(img, sp, sr)

    # === 3. DBSCAN (OpenCV-подобная реализация) ===
    elif method == "dbscan":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3,3), np.uint8)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

        num_labels, labels = cv2.connectedComponents(bin_img)

        result = np.zeros_like(img)
        for lab in range(1, num_labels):
            mask = labels == lab
            result[mask] = img[mask].mean(axis=0)

    # === 4. Snake Segmentation (OpenCV approximation) ===
    elif method == "snake":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        result = img.copy()
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(result, [cnt], -1, (0,255,0), 2)

    else:
        return jsonify(error="Неизвестный алгоритм"), 400

    cv2.imwrite(UPLOAD_PATH, result)
    return jsonify(message="Сегментация выполнена")




if __name__ == "__main__":
    app.run(debug=True)
