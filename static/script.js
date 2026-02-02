const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function loadToCanvas() {
    const img = new Image();
    img.src = "/image?" + Date.now();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
}

function applyMask() {
    let mask = canvas.toDataURL("image/png");

    fetch("/mask_crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}

function refresh() {
    loadToCanvas();
}

function upload() {
    let file = document.getElementById("file").files[0];
    let form = new FormData();
    form.append("image", file);

    fetch("/upload", { method: "POST", body: form })
        .then(r => r.json())
        .then(d => {
            if (d.error) alert(d.error);
            else refresh();
        });
}

function resize() {
    fetch("/resize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            fx: parseFloat(fx.value),
            fy: parseFloat(fy.value)
        })
    }).then(refresh);
}

function crop() {
    fetch("/crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            x: +x.value,
            y: +y.value,
            w: +w.value,
            h: +h.value
        })
    }).then(r => r.json()).then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}

function flip(mode) {
    fetch("/flip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode })
    }).then(refresh);
}

function rotate() {
    fetch("/rotate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ angle: +angle.value })
    }).then(refresh);
}

function brightness() {
    fetch("/brightness", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            contrast: +alpha.value,
            brightness: +beta.value
        })
    }).then(refresh);
}

function color() {
    fetch("/color", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            r: +r.value,
            g: +g.value,
            b: +b.value
        })
    }).then(refresh);
}

function applyNoise() {
    fetch("/noise", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            type: noiseType.value,
            amount: +noiseAmount.value
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}


function blurPhoto() {
    console.log("hello")
    fetch("/blur", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            type: blurType.value,
            kernel: +kernel.value
        })
    })
        .then(r => r.json())
        .then(d => {
            if (d.error) alert(d.error);
            else refresh();
        });
}

function saveImage() {
    fetch("/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            format: format.value,
            quality: quality.value
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(d => alert(d.error));
        }
        return response.blob();
    })
    .then(blob => {
        let url = window.URL.createObjectURL(blob);
        let a = document.createElement("a");
        a.href = url;
        a.download = "result." + format.value;
        a.click();
        window.URL.revokeObjectURL(url);
    });
}

function toggleQuality() {
    const format = document.getElementById("format").value;
    const qualityBlock = document.getElementById("qualityBlock");

    if (format === "jpg" || format === "jpeg") {
        qualityBlock.style.display = "block";
    } else {
        qualityBlock.style.display = "none";
    }
}

function undo() {
    fetch("/undo", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            if (d.error) alert(d.error);
            else refresh();
        });
}

function resetImage() {
    fetch("/reset", { method: "POST" })
        .then(r => r.json())
        .then(d => {
            if (d.error) alert(d.error);
            else refresh();
        });
}

function changeColorSpace() {
    fetch("/colorspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            space: colorSpace.value
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}

function findObject() {
    const hex = objectColor.value;
    const r = parseInt(hex.substr(1,2),16);
    const g = parseInt(hex.substr(3,2),16);
    const b = parseInt(hex.substr(5,2),16);

    fetch("/find_object", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            color: [r, g, b],
            space: searchSpace.value,
            tolerance: tolerance.value,
            mode: searchMode.value
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}

function toggleEdgeParams() {
    const isSobel = edgeMethod.value === "sobel";
    sobelParams.style.display = isSobel ? "block" : "none";
    cannyParams.style.display = isSobel ? "none" : "block";
}

function applyEdges() {
    fetch("/edges", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            method: edgeMethod.value,
            ksize: ksize?.value,
            t1: t1?.value,
            t2: t2?.value
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}

function toggleSegParams() {
    kmeansParams.style.display =
        segMethod.value === "kmeans" ? "block" : "none";

    meanshiftParams.style.display =
        segMethod.value === "meanshift" ? "block" : "none";
}

function applySegmentation() {
    fetch("/segment", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            method: segMethod.value,
            k: k?.value,
            sp: sp?.value,
            sr: sr?.value
        })
    })
    .then(r => r.json())
    .then(d => {
        if (d.error) alert(d.error);
        else refresh();
    });
}





setInterval(() => {
    refresh();
}, 1000);
