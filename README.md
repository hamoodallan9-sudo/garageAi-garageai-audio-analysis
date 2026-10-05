# GarageAI — Audio Analysis Microservice

سيرفر FastAPI داخلي (مش مواجه للمستخدم مباشرة) بيوفر خدمتين لباقي مشروع GarageAI:

1. **تحليل صوت العطل** (`POST /predict`) — بياخد ملف صوت، يشغّله عبر Ensemble من 4 موديلات (Traditional ML, CNN, YAMNet, PANNs) ويرجع التشخيص.
2. **تشخيص نصي بالمحادثة** (`POST /diagnose/text`) — RAG chatbot: بياخد محادثة نصية بالعامية، يطابقها دلاليًا مع قاعدة معرفة أعطال (`app/data/car_faults_glossary.md`) عبر embeddings محلية، وبيولّد رد عبر Google Gemini مبني على السياق المسترجع بس.

بيتنادى من `garageai-backend` (Node.js API gateway) — مش من الفرونت اند مباشرة.

## البنية

```
garageai-audio-analysis/
├── app/
│   ├── main.py                  # FastAPI app + startup (تحميل الموديلات + فهرس RAG)
│   ├── core/config.py           # الإعدادات (.env)
│   ├── routes/
│   │   ├── predict.py           # POST /predict
│   │   └── diagnose.py          # POST /diagnose/text
│   ├── schemas/                 # Pydantic request/response models
│   ├── services/
│   │   ├── model_loader.py      # تحميل الموديلات المدرَّبة (.pkl / .keras) من MODELS_DIR
│   │   ├── feature_extraction.py# استخراج الـ MFCC/log-mel (منسوخ حرفيًا من car_test)
│   │   ├── ensemble.py          # دمج تنبؤات الموديلات الأربعة
│   │   ├── cnn_custom_layers.py # طبقة Keras مخصصة لازمة لتحميل cnn_model.keras
│   │   ├── audio_converter.py   # تحويل أي صيغة صوت لـ WAV عبر ffmpeg
│   │   └── rag/
│   │       ├── ingest.py        # يحلل ملف الـ glossary (Markdown) لقطع (chunks)
│   │       ├── store.py         # embeddings محلية (sentence-transformers) + بحث تشابه
│   │       └── diagnose.py      # يبني الـ prompt ويستدعي Gemini
│   └── data/car_faults_glossary.md  # قاعدة معرفة الأعطال (مصدر الـ RAG)
├── requirements.txt
└── .env                          # غير مرفوع على git (.gitignore)
```

## ملاحظة مهمة: هاد المشروع مش مستقل بالكامل

الموديلات المدرَّبة (`MODELS_DIR`) بتيجي من مشروع تدريب منفصل هو **`car_test`** (سكريبتات `preprocessing.py` + `train_*.py`). هاد المشروع (`garageai-audio-analysis`) بس بيحمّل مخرجات `car_test` ويعرضها كـ API — ما بيدرّب ولا بيعالج بيانات خام. لو تغيّرت الـ features بـ `car_test/preprocessing.py`، لازم `app/services/feature_extraction.py` هون ينسخ نفس التغيير يدويًا (موثّق بتعليق داخل الملف نفسه)، وإلا بترجع أخطاء "feature count mismatch".

## التشغيل

### 1) تجهيز البيئة (مرة وحدة)

```powershell
cd C:\Users\hp\Desktop\garageai-audio-analysis
python -m venv venv          # لو مش موجود أصلاً
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) متغيرات البيئة (`.env`)

| المتغير | الوصف | إلزامي؟ |
|---|---|---|
| `MODELS_DIR` | مسار مجلد `processed_data` الناتج من `car_test` | نعم — لازم يطابق فعليًا مكان الموديلات على جهازك |
| `GEMINI_API_KEY` | مفتاح Google Gemini API لتوليد ردود `/diagnose/text` | بدونه، الـ endpoint بيشتغل بس بيرجع رسالة إنه الذكاء الاصطناعي غير مفعّل |
| `RAG_MATCH_THRESHOLD` | حد أدنى لدرجة التشابه (0-1) قبل ما نعتبر إنه فيه تطابق فعلي مع قاعدة المعرفة | لأ — افتراضي `0.35` |

### 3) التشغيل

```powershell
.\venv\Scripts\python.exe -m app.main
```

بيشتغل على `http://127.0.0.1:8001` (لازم يطابق `PYTHON_SERVICE_URL` بـ `.env` تبع `garageai-backend`، ويفضّل `127.0.0.1` مش `localhost` — تجنبًا لمشكلة IPv6/IPv4 على Windows).

Swagger docs تفاعلية: `http://127.0.0.1:8001/docs`

### أول تشغيل بياخد وقت أطول

تحميل TensorFlow + YAMNet + sentence-transformers أول مرة بياخد دقيقة لدقيقتين (خصوصًا لو YAMNet لازم يتحمّل من الإنترنت أول مرة). بعد أول تشغيل، بينحفظ بالكاش المحلي وبيصير أسرع بكثير.

### 4) التشغيل عبر Docker (بديل)

```bash
docker build -t garageai-audio-analysis .
docker run -p 8001:8001 --env-file .env \
  -v /path/to/car_test/processed_data:/models -e MODELS_DIR=/models \
  garageai-audio-analysis
```

## الاعتماديات الخارجية

- **car_test** (`C:\Users\hp\Desktop\car_test`) — مصدر الموديلات المدرَّبة (`MODELS_DIR`)
- **garageai-backend** — بينادي `/predict` و `/diagnose/text` عبر HTTP، وهو نقطة الدخول الوحيدة لهاد السيرفر (ما في CORS هون لأنه ما بيستقبل طلبات مباشرة من متصفح)
