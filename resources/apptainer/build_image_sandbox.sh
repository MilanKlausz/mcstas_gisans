#!/bin/bash
# Build a SIF image from a .def file (Bootstrap: docker) WITHOUT root, e.g. on the DMSC cluster
# login node (singularity 3.5.3), where
#   - "singularity build x.sif x.def" needs root, and --fakeroot fails (no /etc/subuid entry);
#   - /usr/bin/apptainer is broken on the login node (undefined symbol seccomp_notify_respond),
#     hence /usr/local/bin/singularity (override with SINGULARITY=<path>);
#   - singularity 3.5.3 cannot read the multi-arch OCI index of docker://python:3.11
#     ("unsupported schema version 2"), so the base image (the "From:" line of the def file) is
#     pulled by its linux/amd64 digest.
# Steps: sandbox from docker (files owned by the user) -> run the def's %post section in the
# writable sandbox -> pip freeze to <out>_pip_freeze.txt -> run the %test section -> store the def
# inside the image (/.singularity.d/Singularity.def_source) -> squash the sandbox into a SIF.
# Only the %post and %test sections are used (no %files, %environment, %runscript, ...).
# With root, simply use: sudo singularity build <out.sif> <file.def>
# Usage: build_image_sandbox.sh <file.def> <out.sif>
set -euo pipefail
DEF=$(readlink -f "$1"); OUT=$(readlink -f "$2")
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT"; exit 1; }
SING=${SINGULARITY:-/usr/local/bin/singularity}
WORK=$(mktemp -d /tmp/${USER}_imgbuild.XXXXXX)
export SINGULARITY_CACHEDIR=$WORK/cache SINGULARITY_TMPDIR=$WORK/tmp
mkdir -p "$SINGULARITY_TMPDIR" "$WORK/wd"
# --workdir: /tmp inside the container on disk (with --contain it is otherwise a 64 MB session dir)
OPTS=(--contain --no-home --workdir "$WORK/wd")

# linux/amd64 manifest digest of the base image (Docker Hub), e.g. "From: python:3.11"
FROM=$(awk '/^From:/{print $2; exit}' "$DEF")
REPO=${FROM%%:*}; TAG=${FROM#*:}; [ "$TAG" = "$FROM" ] && TAG=latest
case "$REPO" in */*) ;; *) REPO=library/$REPO;; esac
TOKEN=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
DIGEST=$(curl -s -H "Authorization: Bearer $TOKEN" -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json" \
  "https://registry-1.docker.io/v2/$REPO/manifests/$TAG" | python3 -c "
import sys,json
for m in json.load(sys.stdin)['manifests']:
    p=m.get('platform',{})
    if p.get('architecture')=='amd64' and p.get('os')=='linux': print(m['digest']); break")
[ -n "$DIGEST" ] || { echo "no linux/amd64 digest found for $FROM"; exit 1; }
echo "$FROM linux/amd64 digest: $DIGEST"

SB=$WORK/sandbox
$SING build --sandbox "$SB" "docker://${REPO#library/}@$DIGEST"
# %post section of the def file (lines after %post up to the next section), run in the writable sandbox
POST=$(awk '/^%post/{f=1;next} /^%/{f=0} f' "$DEF")
$SING exec --writable "${OPTS[@]}" "$SB" /bin/bash -euxc "$POST"
$SING exec "${OPTS[@]}" "$SB" python -m pip freeze > "${OUT%.sif}_pip_freeze.txt"
# %test section
TEST=$(awk '/^%test/{f=1;next} /^%/{f=0} f' "$DEF")
$SING exec "${OPTS[@]}" "$SB" /bin/bash -euxc "$TEST"
# record the definition file and the base digest in the image as /.singularity.d/Singularity.def_source
# (the sandbox->SIF build overwrites /.singularity.d/Singularity, i.e. "inspect --deffile" shows only "from: <sandbox>")
{ cat "$DEF"; echo; echo "# built with build_image_sandbox.sh from docker://${REPO#library/}@$DIGEST on $(date -I)"; } > "$SB/.singularity.d/Singularity.def_source"
rm -rf "$SB/root/.cache" "$SB/tmp/"* || true
$SING build "$OUT" "$SB"
rm -rf "$WORK"
ls -la "$OUT"
