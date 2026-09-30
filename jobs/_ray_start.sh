# Start Ray before RLinf does, with retries, on a per-job port and temp dir.
# RLinf's Cluster first tries ray.init(address="auto") and only then starts its
# own. Two things this guards against:
#   * Ray's local startup has a hard-coded 30 s raylet wait (job 54408137 died
#     on "The current node timed out during startup").
#   * On the shared partition two jobs can land on one node; both used port
#     6379 and /tmp/ray, and the second failed to reach GCS (job 54460480).
# Source this after _common.sh so Ray captures the environment it exports.
export RLINF_NODE_RANK=0   # RLinf reads it from the environment Ray captured at start
# Every Ray port (workers included) gets a per-job block of 1000: GCS alone was not enough, two
# evals on one node still collided on the client-server port 10002 (54464953).
RAY_PORT=$((10000 + (${SLURM_JOB_ID:-0} % 50) * 1000))
RAY_TMP=/tmp/ray_${SLURM_JOB_ID:-$$}
mkdir -p $RAY_TMP
# Never `ray stop`: it kills every Ray process of this user on the node,
# including another job's on a shared node. Match this job's temp dir instead.
ray_kill_own() { pkill -u "$USER" -f -- "--temp-dir=$RAY_TMP" 2>/dev/null; pkill -u "$USER" -f -- "$RAY_TMP/session" 2>/dev/null; true; }
for attempt in 1 2 3; do
    if ray start --head --port=$RAY_PORT --temp-dir=$RAY_TMP --disable-usage-stats \
            --include-dashboard=false --ray-client-server-port=$((RAY_PORT + 1)) \
            --node-manager-port=$((RAY_PORT + 2)) --object-manager-port=$((RAY_PORT + 3)) \
            --metrics-export-port=$((RAY_PORT + 4)) --dashboard-agent-grpc-port=$((RAY_PORT + 5)) \
            --dashboard-agent-listen-port=$((RAY_PORT + 6)) --runtime-env-agent-port=$((RAY_PORT + 7)) \
            --min-worker-port=$((RAY_PORT + 100)) --max-worker-port=$((RAY_PORT + 999)) \
            >/dev/null 2>&1 \
        && ray status --address=127.0.0.1:$RAY_PORT >/dev/null 2>&1; then
        echo "ray up (attempt $attempt) on port $RAY_PORT"; break
    fi
    echo "ray start failed (attempt $attempt); retrying"
    ray_kill_own; sleep 30
    [ $attempt -eq 3 ] && { echo "FATAL: ray would not start"; exit 1; }
done
export RAY_ADDRESS=127.0.0.1:$RAY_PORT
trap 'ray_kill_own; rm -rf $RAY_TMP' EXIT
