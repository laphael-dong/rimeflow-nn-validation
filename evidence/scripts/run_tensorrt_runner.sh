#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
image_name=rimeflow-tensorrt-ep:ort-1.22.0
record=false
if [[ " ${*:-} " == *" --record "* ]]; then
  record=true
fi
cd "$repo_root"
workspace="$repo_root/.evidence/tensorrt/container-run"
mkdir -p "$workspace"

publish_candidate() {
  local candidate=$1
  node "$repo_root/evidence/scripts/validate_tensorrt_ep.mjs" "$candidate" "$workspace"
  if $record; then
    python3 "$repo_root/evidence/scripts/tensorrt_record_publish.py" \
      --candidate "$candidate" \
      --target "$repo_root/evidence/reports/tensorrt-ep-report.json" \
      --validator "$repo_root/evidence/scripts/validate_tensorrt_ep.mjs"
  fi
}

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
  set +e
  python3 "$repo_root/evidence/scripts/run_tensorrt_ep.py" --workspace "$workspace"
  runner_exit=$?
  set -e
  [[ $runner_exit -eq 2 ]]
  publish_candidate "$workspace/tensorrt-ep-report.json"
  exit "$runner_exit"
fi

verify_image() {
  local image_ref=$1
  local expected_index=$2
  local expected_amd64=$3
  local inspection
  inspection=$(docker buildx imagetools inspect "$image_ref")
  grep -Fq "Digest:    $expected_index" <<<"$inspection"
  grep -Fq "@$expected_amd64" <<<"$inspection"
}

verify_image rust:1.85.1-alpine3.21 \
  sha256:4333721398de61f53ccbe53b0b855bcc4bb49e55828e8f652d7a8ac33dd0c118 \
  sha256:813376c206852d4250641eb86720c04fd20fbc6547d1a030c93a45b22a1303a3
verify_image nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04 \
  sha256:c900a2ac7f3d860c854f7364eadfe32b8cd33933a838f96b71e9bb6525f77a8c \
  sha256:2a015be069bda4de48d677b6e3f271a2794560c7d788a39a18ecf218cae0751d

bun install --frozen-lockfile --cwd "$repo_root/evidence/tooling/web"
node "$repo_root/evidence/scripts/export_web_reference_tensors.mjs" "$workspace/web-reference"

docker build --pull --platform linux/amd64 --tag "$image_name" --file "$repo_root/evidence/tooling/tensorrt/Dockerfile" "$repo_root"
image_id=$(docker image inspect "$image_name" --format '{{.Id}}')
mkdir -p "$workspace/runtime"
docker image inspect "$image_name" > "$workspace/runtime/container-image-inspect.json"
set +e
docker run --rm --gpus all \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --volume "$repo_root:/workspace:ro" \
  --volume "$repo_root/.evidence/tensorrt:/workspace/.evidence/tensorrt:rw" \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env "RIMEFLOW_TENSORRT_IMAGE_ID=$image_id" \
  "$image_name" \
  python3.10 evidence/scripts/run_tensorrt_ep.py --workspace .evidence/tensorrt/container-run
runner_exit=$?
set -e
candidate="$workspace/tensorrt-ep-report.json"
publish_candidate "$candidate"
exit "$runner_exit"
