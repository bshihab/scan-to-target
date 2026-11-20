#!/usr/bin/env bash
# Static copy of the planner (no server) for a website: viewer + one scene.
# Usage: scripts/build_static_demo.sh [scene_dir] [dest]   (defaults: out/scenes/sub-01_ses-test dist/demo)
# Export downloads plan.json in the browser instead of posting it. robot_demo.json is not copied.
set -euo pipefail
cd "$(dirname "$0")/.."
SCENE=${1:-out/scenes/sub-01_ses-test}
DEST=${2:-dist/demo}
# only ever replace a folder this script made (it leaves a marker file), never e.g. out/
if [ -e "$DEST" ] && [ ! -f "$DEST/.s2t-static-demo" ]; then
  echo "refusing to replace $DEST: it exists and was not made by this script" >&2
  exit 1
fi
rm -rf "$DEST" && mkdir -p "$DEST/scene" && touch "$DEST/.s2t-static-demo"
cp s2t/viewer/index.html s2t/viewer/app.js s2t/viewer/geometry.js s2t/viewer/style.css "$DEST/"
cp "$SCENE/scene.json" "$SCENE/t1.u8.gz" "$SCENE/overlay.f32.gz" "$DEST/scene/"
du -sh "$DEST"
