[app]
title = NavApp
package.name = navapp
package.domain = org.navapp
source.dir = .
source.include_exts = py,pt,wav
version = 1.0

requirements = python3,kivy,plyer,numpy,torch,ultralytics,opencv

android.permissions = CAMERA,VIBRATE,RECORD_AUDIO
android.api = 33
android.minapi = 21
android.archs = arm64-v8a
orientation = portrait

[buildozer]
log_level = 2
```

---

## Your Folder Structure in VS Code
```
navapp/
├── main.py          ← paste the converted code above
├── buildozer.spec   ← paste the spec above
└── yolov8n.pt       ← copy from your existing project