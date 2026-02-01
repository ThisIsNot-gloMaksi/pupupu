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
            contrast: +contrast.value,
            brightness: +brightness.value
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

function noise() {
    fetch("/noise", { method: "POST" }).then(refresh);
}

function blur() {
    fetch("/blur", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kernel: +kernel.value })
    }).then(refresh);
}

setInterval(() => {
    refresh();
}, 1000);
