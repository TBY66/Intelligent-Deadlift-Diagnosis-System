#!/bin/bash
# build_app.sh
# Creates "Deadlift Diagnosis.app" in the project folder.
# The .app is a lightweight wrapper — it calls your conda Python directly,
# so all models and dependencies work without re-bundling them.
set -e
cd "$(dirname "$0")"

PYTHON="/opt/anaconda3/bin/python"
PROJECT="$(pwd)"
APP_NAME="Deadlift Diagnosis"
APP="${PROJECT}/${APP_NAME}.app"
PNG="${PROJECT}/materials/icon1_blue.png"
ICNS="${PROJECT}/AppIcon.icns"

# ── 1. PNG → ICNS ─────────────────────────────────────────────────────────────
echo "Converting icon…"
ICONSET="${PROJECT}/AppIcon.iconset"
rm -rf "$ICONSET" && mkdir -p "$ICONSET"

sips -z 16   16   "$PNG" --out "${ICONSET}/icon_16x16.png"       &>/dev/null
sips -z 32   32   "$PNG" --out "${ICONSET}/icon_16x16@2x.png"    &>/dev/null
sips -z 32   32   "$PNG" --out "${ICONSET}/icon_32x32.png"        &>/dev/null
sips -z 64   64   "$PNG" --out "${ICONSET}/icon_32x32@2x.png"    &>/dev/null
sips -z 128  128  "$PNG" --out "${ICONSET}/icon_128x128.png"      &>/dev/null
sips -z 256  256  "$PNG" --out "${ICONSET}/icon_128x128@2x.png"  &>/dev/null
sips -z 256  256  "$PNG" --out "${ICONSET}/icon_256x256.png"      &>/dev/null
sips -z 512  512  "$PNG" --out "${ICONSET}/icon_256x256@2x.png"  &>/dev/null
sips -z 512  512  "$PNG" --out "${ICONSET}/icon_512x512.png"      &>/dev/null
sips -z 1024 1024 "$PNG" --out "${ICONSET}/icon_512x512@2x.png"  &>/dev/null

iconutil -c icns "$ICONSET" -o "$ICNS"
rm -rf "$ICONSET"
echo "  ✓ AppIcon.icns"

# ── 2. App bundle skeleton ─────────────────────────────────────────────────────
echo "Building ${APP_NAME}.app…"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
mkdir -p "$APP/Contents/Resources"
cp "$ICNS" "$APP/Contents/Resources/AppIcon.icns"

# ── 3. Launcher script (the real executable inside the bundle) ─────────────────
# Logs to ~/Library/Logs/DeadliftDiagnosis.log so crashes are visible.
cat > "$APP/Contents/MacOS/launcher" <<LAUNCHER
#!/bin/bash
cd "${PROJECT}"
exec "${PYTHON}" "${PROJECT}/appv1.2.py" >"\${HOME}/Library/Logs/DeadliftDiagnosis.log" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/launcher"

# ── 4. Info.plist ──────────────────────────────────────────────────────────────
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>               <string>Deadlift Diagnosis</string>
  <key>CFBundleDisplayName</key>        <string>Intelligent Deadlift Diagnosis</string>
  <key>CFBundleIdentifier</key>         <string>com.deadlift.diagnosis</string>
  <key>CFBundleVersion</key>            <string>1.2.0</string>
  <key>CFBundleShortVersionString</key> <string>1.2</string>
  <key>CFBundleExecutable</key>         <string>launcher</string>
  <key>CFBundleIconFile</key>           <string>AppIcon</string>
  <key>CFBundlePackageType</key>        <string>APPL</string>
  <key>NSHighResolutionCapable</key>    <true/>
  <key>NSRequiresAquaSystemAppearance</key> <false/>
  <key>LSMinimumSystemVersion</key>     <string>12.0</string>
</dict>
</plist>
PLIST

# Tell Finder to refresh the icon cache
touch "$APP"

echo "  ✓ ${APP_NAME}.app created in:"
echo "    ${APP}"
echo ""
echo "First launch: right-click the .app → Open  (bypasses Gatekeeper once)."
echo "Crash log:    ~/Library/Logs/DeadliftDiagnosis.log"
