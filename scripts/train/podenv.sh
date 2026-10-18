# Source this in any new terminal to talk to the pods:
#
#     source scripts/train/podenv.sh
#
# Defines podssh / cpussh and loads .env.runpod. Both are shell FUNCTIONS, not
# scripts, so they exist only in the shell that sourced this file — a new
# terminal tab needs to source it again. That is also why they are functions and
# not a POD="-p ... -i ..." variable: in bash an unquoted $POD word-splits into
# separate arguments, but zsh passes it as ONE argument and ssh then reads
# everything after -p as the port ("Bad port"). A function passes "$@" through
# cleanly in both shells. scripts/train/runpod_wizard.sh:246 uses the same shape.

_here="${${(%):-%x}:A:h:h:h}"       # zsh: this file's repo root
[ -n "$_here" ] || _here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

set -a
# shellcheck disable=SC1090
. "$_here/.env.runpod"
set +a
unset _here

podssh() {
  ssh -p "$GPU_POD_PORT" -i "$SSH_KEY" \
      -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      root@"$GPU_POD_HOST" "$@"
}

cpussh() {
  ssh -p "$CPU_POD_PORT" -i "$SSH_KEY" \
      -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 \
      root@"$CPU_POD_HOST" "$@"
}

printf 'pod env loaded — GPU %s:%s\n' "$GPU_POD_HOST" "$GPU_POD_PORT"
printf '  podssh "<command>"   run on the GPU pod\n'
printf '  cpussh "<command>"   run on the CPU pod\n'
[ -n "$COMET_API_KEY" ] \
  && printf '  COMET_API_KEY loaded (%d chars)\n' "${#COMET_API_KEY}" \
  || printf '  COMET_API_KEY NOT set — training will log to TensorBoard only\n'
