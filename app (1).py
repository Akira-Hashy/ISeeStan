import os
import re

import requests
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024  # istek gövdesi en fazla 6 MB

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

PROMPTS = {
    "gorme": (
        "Sen görme engelli kullanıcılara yardım eden bir asistansın. "
        "Fotoğrafları Türkçe, sade ve sıralı anlat: önce ana nesne veya sahne, sonra konumlar "
        "(sol, sağ, ön, arka), renkler, varsa yazılar. Yazı veya sayı görürsen aynen oku. "
        "Tehlike varsa (merdiven, trafik, keskin nesne) en başta söyle. "
        "Emin olmadığın şeyi 'sanki' veya 'olabilir' diyerek belirt, uydurma. "
        "Yanıtı sesli okunacak: madde işareti, emoji, yıldız veya tablo kullanma, kısa cümleler kur."
    ),
    "genel": (
        "Sen Türkçe konuşan yardımcı bir asistansın. Açık ve doğru cevap ver. "
        "Fotoğrafta matematiksel sembol varsa sembolü adıyla tanıt, ne anlama geldiğini ve "
        "hangi alanda kullanıldığını kısaca açıkla. Emin değilsen bunu söyle."
    ),
}


def clean(text: str) -> str:
    # Bazı modeller <think>...</think> düşünce bloğunu cevaba karıştırır.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def err(message: str, status: int):
    return jsonify(error=message), status


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.errorhandler(413)
def too_large(_):
    return err("Fotoğraf çok büyük. Daha küçük bir fotoğraf dene.", 413)


@app.post("/api/chat")
def chat():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return err("Sunucuda GROQ_API_KEY tanımlı değil. Render > Environment bölümüne ekle.", 500)

    data = request.get_json(silent=True) or {}
    mode = data.get("mode") if data.get("mode") in PROMPTS else "genel"
    text = str(data.get("text") or "").strip()[:4000]
    image = data.get("image")

    if not text and not image:
        return err("Mesaj boş.", 400)
    if image is not None and not (isinstance(image, str) and image.startswith("data:image/")):
        return err("Fotoğraf biçimi geçersiz.", 400)

    messages = [{"role": "system", "content": PROMPTS[mode]}]
    for m in (data.get("history") or [])[-10:]:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            messages.append({"role": m["role"], "content": m["content"][:4000]})

    if image:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text or "Bu fotoğrafı anlat."},
                {"type": "image_url", "image_url": {"url": image}},
            ],
        })
    else:
        messages.append({"role": "user", "content": text})

    try:
        r = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": MODEL, "messages": messages, "temperature": 0.6, "max_tokens": 1200},
            timeout=90,
        )
    except requests.Timeout:
        return err("Yapay zeka zamanında cevap vermedi. Tekrar dene.", 504)
    except requests.RequestException:
        return err("Yapay zeka servisine ulaşılamadı.", 502)

    if r.status_code != 200:
        try:
            detail = r.json()["error"]["message"]
        except Exception:
            detail = r.text[:200]
        if r.status_code == 429:
            return err("İstek limiti doldu. Biraz bekleyip tekrar dene.", 429)
        return err(f"API hatası ({r.status_code}): {detail}", 502)

    try:
        reply = clean(r.json()["choices"][0]["message"]["content"] or "")
    except Exception:
        return err("API'den beklenmeyen bir cevap geldi.", 502)

    return jsonify(reply=reply or "Cevap boş geldi, tekrar dene.")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
