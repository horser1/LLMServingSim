import json
import random
from .logger import get_logger


class Router:
    def __init__(
            self,
            num_instances,
            schedulers, req_num,
            routing_policy="RR",
            seed=42,
            pools=None,
    ):
        self.schedulers = schedulers
        self.scheduler_by_id = {s.instance_id: s for s in schedulers}
        self.num_instances = num_instances
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self._selection_counters = {}

        # Pending requests (loaded but not yet routed)
        self._pending_requests = []
        self._pending_idx = 0
        self._enable_prefix_caching = False
        self._is_init = True

        # Agentic session dependency tracking
        self._deferred_sessions = {}     # session_id -> session state dict
        self._request_to_session = {}    # request_id -> (session_id, sub_request_index)
        self._next_request_id = 0        # monotonic counter for unique request IDs

        if self.routing_policy == "RR":
            self._select_instance = self._rr_select
        elif self.routing_policy == "RAND":
            self._select_instance = self._rand_select
        elif self.routing_policy == "LOAD":
            self._select_instance = self._least_load_select
        elif self.routing_policy == "CUSTOM":
            self._select_instance = self._custom_select
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, LOAD, CUSTOM")

        self.pools = self._build_pool_states(pools)
        self.pool_by_id = {p["id"]: p for p in self.pools}
        self.instance_pool = {}
        for pool in self.pools:
            for inst_id in pool["instances"]:
                self.instance_pool[inst_id] = pool["id"]

        self.logger = get_logger(self.__class__)

    # -----------------------------------------------------------------------
    # Pool setup
    # -----------------------------------------------------------------------

    def _legacy_pools_from_schedulers(self):
        agg_instances = [s.instance_id for s in self.schedulers if s.pd_type is None]
        prefill_instances = [s.instance_id for s in self.schedulers if s.pd_type == "prefill"]
        decode_instances = [s.instance_id for s in self.schedulers if s.pd_type == "decode"]

        if agg_instances and not prefill_instances and not decode_instances:
            return [{
                "id": "default",
                "mode": "agg",
                "instances": agg_instances,
                "prefill_instances": [],
                "decode_instances": [],
                "admission": {},
                "fallback": [],
            }]
        if not agg_instances and prefill_instances and decode_instances:
            return [{
                "id": "default",
                "mode": "pd",
                "instances": prefill_instances + decode_instances,
                "prefill_instances": prefill_instances,
                "decode_instances": decode_instances,
                "admission": {},
                "fallback": [],
            }]
        raise ValueError("Mixed agg and P/D routing requires explicit pools.")

    def _build_pool_states(self, pools):
        pools = pools if pools is not None else self._legacy_pools_from_schedulers()
        states = []
        for pool_cfg in pools:
            pool = dict(pool_cfg)
            pool.setdefault("admission", {})
            pool.setdefault("fallback", [])
            pool.setdefault("prefill_instances", [])
            pool.setdefault("decode_instances", [])
            if pool["mode"] == "agg":
                pool["agg_schedulers"] = self._schedulers_for_ids(pool["instances"], pool["id"], "agg")
                pool["prefill_schedulers"] = []
                pool["decode_schedulers"] = []
            elif pool["mode"] == "pd":
                pool["agg_schedulers"] = []
                pool["prefill_schedulers"] = self._schedulers_for_ids(
                    pool["prefill_instances"], pool["id"], "prefill")
                pool["decode_schedulers"] = self._schedulers_for_ids(
                    pool["decode_instances"], pool["id"], "decode")
            else:
                raise ValueError(f"Unknown pool mode '{pool['mode']}' for pool '{pool['id']}'.")
            states.append(pool)
        return states

    def _schedulers_for_ids(self, instance_ids, pool_id, role):
        schedulers = []
        for inst_id in instance_ids:
            if inst_id not in self.scheduler_by_id:
                raise ValueError(f"Pool '{pool_id}' references unknown instance {inst_id}.")
            schedulers.append(self.scheduler_by_id[inst_id])
        if not schedulers:
            raise ValueError(f"Pool '{pool_id}' has no {role} schedulers.")
        return schedulers

    # -----------------------------------------------------------------------
    # Instance selection policies
    # -----------------------------------------------------------------------

    def _get_counter(self, counter_key):
        return self._selection_counters.get(counter_key, 0)

    def _set_counter(self, counter_key, value):
        self._selection_counters[counter_key] = value

    def _rr_select(self, schedulers, counter_key):
        num_instances = len(schedulers)
        if num_instances == 0:
            raise RuntimeError("No scheduler candidates are available for routing.")
        idx = self._get_counter(counter_key) % num_instances
        self._set_counter(counter_key, idx + 1)
        return idx

    def _rand_select(self, schedulers, counter_key):
        num_instances = len(schedulers)
        if num_instances == 0:
            raise RuntimeError("No scheduler candidates are available for routing.")
        return self._rnd.randrange(num_instances)

    def _least_load_select(self, schedulers, counter_key):
        """vLLM-style least-loaded routing, normalized by instance capacity."""
        num_instances = len(schedulers)
        if num_instances == 0:
            raise RuntimeError("No scheduler candidates are available for routing.")
        best_idx = 0
        best_score = float('inf')
        start = self._get_counter(counter_key) % num_instances
        for offset in range(num_instances):
            idx = (start + offset) % num_instances
            sched = schedulers[idx]
            waiting = len(sched.request)
            running = sum(len(b.requests) for b in sched.inflight)
            raw_score = waiting * 4 + running
            capacity = getattr(sched, "max_num_seqs", 0)
            score = raw_score
            if capacity not in (0, float('inf')):
                score = raw_score / capacity
            if score < best_score:
                best_score = score
                best_idx = idx
        self._set_counter(counter_key, (best_idx + 1) % num_instances)
        return best_idx

    def _custom_select(self, schedulers, counter_key):
        raise NotImplementedError("Implement custom routing policy.")

    # -----------------------------------------------------------------------
    # Pool admission and fallback
    # -----------------------------------------------------------------------

    def _token_metrics(self, req_data):
        input_toks = int(req_data["input_toks"])
        total_toks = int(req_data["output_toks"])
        output_toks = max(0, total_toks - input_toks)
        return {
            "input_toks": input_toks,
            "output_toks": output_toks,
            "total_toks": total_toks,
        }

    def _token_admission_passes(self, pool, req_data):
        admission = pool.get("admission") or {}
        metrics = self._token_metrics(req_data)
        checks = (
            ("min_input_toks", "input_toks", lambda actual, limit: actual >= limit),
            ("max_input_toks", "input_toks", lambda actual, limit: actual <= limit),
            ("min_output_toks", "output_toks", lambda actual, limit: actual >= limit),
            ("max_output_toks", "output_toks", lambda actual, limit: actual <= limit),
            ("min_total_toks", "total_toks", lambda actual, limit: actual >= limit),
            ("max_total_toks", "total_toks", lambda actual, limit: actual <= limit),
        )
        for key, metric, predicate in checks:
            if key in admission and not predicate(metrics[metric], admission[key]):
                return False
        return True

    def _initial_role(self, pool):
        return "agg" if pool["mode"] == "agg" else "prefill"

    def _schedulers_for_role(self, pool, role):
        if role == "agg":
            return pool["agg_schedulers"]
        if role == "prefill":
            return pool["prefill_schedulers"]
        if role == "decode":
            return pool["decode_schedulers"]
        raise ValueError(f"Unknown routing role '{role}'.")

    def _pool_load(self, pool, role):
        schedulers = self._schedulers_for_role(pool, role)
        waiting = sum(len(s.request) for s in schedulers)
        running = sum(len(b.requests) for s in schedulers for b in s.inflight)
        return waiting, running, waiting * 4 + running

    def _load_admission_passes(self, pool, role):
        admission = pool.get("admission") or {}
        waiting, running, score = self._pool_load(pool, role)
        if "max_waiting" in admission and waiting >= admission["max_waiting"]:
            return False
        if "max_running" in admission and running >= admission["max_running"]:
            return False
        if "max_score" in admission and score >= admission["max_score"]:
            return False
        return True

    def _select_pool(self, req_data):
        for pool in self.pools:
            if not self._token_admission_passes(pool, req_data):
                continue
            role = self._initial_role(pool)
            if self._load_admission_passes(pool, role):
                return pool, None
            fallback_pool = self._select_fallback_pool(pool, req_data)
            if fallback_pool is not None:
                return fallback_pool, pool["id"]
            return None, None
        return None, None

    def _select_fallback_pool(self, source_pool, req_data):
        visited = set()

        def walk(pool_id):
            if pool_id in visited:
                return None
            visited.add(pool_id)
            pool = self.pool_by_id[pool_id]
            role = self._initial_role(pool)
            if self._token_admission_passes(pool, req_data) and self._load_admission_passes(pool, role):
                return pool
            for target in pool.get("fallback", []):
                selected = walk(target)
                if selected is not None:
                    return selected
            return None

        for target in source_pool.get("fallback", []):
            selected = walk(target)
            if selected is not None:
                return selected
        return None

    def _select_pool_scheduler(self, pool, role):
        schedulers = self._schedulers_for_role(pool, role)
        idx = self._select_instance(schedulers, f"{pool['id']}:{role}")
        return schedulers[idx]

    def _route_request_to_scheduler(self, sched, req_data, pool, role, fallback_from):
        route_entry = {
            "pool_id": pool["id"],
            "role": role,
            "instance_id": sched.instance_id,
        }
        route_history = [route_entry]

        if self._enable_prefix_caching:
            sched.add_request([
                req_data["index"], sched.model,
                req_data["input_toks"], req_data["output_toks"],
                req_data["arrival_time_ns"], sched.instance_id,
                req_data.get("input_hash_ids", []), req_data.get("output_hash_ids", []),
            ], is_init=self._is_init, pool_id=pool["id"],
                fallback_from=fallback_from, route_history=route_history)
        else:
            sched.add_request([
                req_data["index"], sched.model,
                req_data["input_toks"], req_data["output_toks"],
                req_data["arrival_time_ns"], sched.instance_id,
            ], is_init=self._is_init, pool_id=pool["id"],
                fallback_from=fallback_from, route_history=route_history)

    # -----------------------------------------------------------------------
    # Request loading and real-time routing
    # -----------------------------------------------------------------------

    def load_requests(self, path, enable_prefix_caching=False, is_init=True):
        """Load requests from dataset into pending queue (not yet routed).

        Supports two JSONL formats:
        - Flat: {"input_toks", "output_toks", "arrival_time_ns", ...}
        - Agentic session: {"session_id", "arrival_time_ns", "sub_requests": [...]}

        For agentic sessions, only the first sub-request is added to the
        pending queue. Subsequent sub-requests are released dynamically
        via notify_request_completed() when predecessors finish.
        """
        path = f'../{path}'
        self._enable_prefix_caching = enable_prefix_caching
        self._is_init = is_init
        loaded_lines = 0

        with open(path) as f:
            for line in f:
                if self.req_num > 0 and loaded_lines >= self.req_num:
                    break
                row = json.loads(line)
                if 'sub_requests' in row:
                    self._load_agentic_session(row, enable_prefix_caching)
                else:
                    self._load_flat_request(row, enable_prefix_caching)
                loaded_lines += 1

        # Sort pending requests by arrival time (agentic first sub-requests
        # may interleave with flat requests)
        self._pending_requests.sort(key=lambda r: r['arrival_time_ns'])

        self.logger.info("Loaded %d requests into pending queue "
                         "(%d agentic sessions deferred)",
                         len(self._pending_requests),
                         len(self._deferred_sessions))

    def _load_flat_request(self, row, enable_prefix_caching):
        """Load a single flat request into pending queue."""
        req_id = self._next_request_id
        self._next_request_id += 1
        req_data = {
            'index': req_id,
            'input_toks': int(row['input_toks']),
            'output_toks': int(row['input_toks'] + row['output_toks']),
            'arrival_time_ns': int(row['arrival_time_ns']),
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = row.get('input_tok_ids', [])
            req_data['output_hash_ids'] = row.get('output_tok_ids', [])
        self._pending_requests.append(req_data)

    def _load_agentic_session(self, row, enable_prefix_caching):
        """Load an agentic session: first sub-request to pending, rest deferred."""
        sub_reqs = row['sub_requests']
        if not sub_reqs:
            return 0
        session_id = row.get('session_id', f'session_{self._next_request_id}')
        base_id = self._next_request_id
        self._next_request_id += len(sub_reqs)
        arrival_ns = int(row['arrival_time_ns'])

        # Store session state for dependency chain
        self._deferred_sessions[session_id] = {
            'sub_requests': sub_reqs,
            'next_index': 1,  # index 0 is being queued now
            'id_base': base_id,
        }

        # Queue the first sub-request
        first = sub_reqs[0]
        req_data = {
            'index': base_id,
            'input_toks': int(first['input_toks']),
            'output_toks': int(first['input_toks'] + first['output_toks']),
            'arrival_time_ns': arrival_ns,
            'session_id': session_id,
            'sub_request_index': 0,
        }
        if enable_prefix_caching:
            req_data['input_hash_ids'] = first.get('input_tok_ids', [])
            req_data['output_hash_ids'] = first.get('output_tok_ids', [])
        self._pending_requests.append(req_data)
        self._request_to_session[base_id] = (session_id, 0)

        return len(sub_reqs)

    def route_arrived_requests(self, current_time_ns):
        """Route requests that have arrived by current_time_ns to pools."""
        routed = 0
        while self._pending_idx < len(self._pending_requests):
            req_data = self._pending_requests[self._pending_idx]
            if req_data['arrival_time_ns'] > current_time_ns:
                break

            pool, fallback_from = self._select_pool(req_data)
            if pool is None:
                break

            role = self._initial_role(pool)
            sched = self._select_pool_scheduler(pool, role)
            self._route_request_to_scheduler(sched, req_data, pool, role, fallback_from)

            self._pending_idx += 1
            routed += 1

        return routed

    def has_pending_requests(self):
        """Check if there are unrouted requests remaining."""
        return self._pending_idx < len(self._pending_requests)

    def get_first_arrival_time(self):
        """Return the first request's arrival time in ns, or 1 if no requests."""
        if self._pending_requests:
            return max(1, self._pending_requests[0]['arrival_time_ns'])
        return 1

    # -----------------------------------------------------------------------
    # Agentic dependency chain management
    # -----------------------------------------------------------------------

    def notify_request_completed(self, request_id, completion_time_ns):
        """Release the next sub-request in a session after tool latency."""
        session_info = self._request_to_session.pop(request_id, None)
        if session_info is None:
            return
        session_id, completed_idx = session_info
        session = self._deferred_sessions.get(session_id)
        if session is None:
            return

        sub_reqs = session['sub_requests']
        next_idx = session['next_index']
        base_id = session['id_base']

        # Get tool duration from the completed sub-request
        tool_duration_ns = int(sub_reqs[completed_idx].get('tool_duration_ns', 0))
        release_time_ns = completion_time_ns + tool_duration_ns

        if next_idx < len(sub_reqs):
            # Release next sub-request
            next_sub = sub_reqs[next_idx]
            next_id = base_id + next_idx
            req_data = {
                'index': next_id,
                'input_toks': int(next_sub['input_toks']),
                'output_toks': int(next_sub['input_toks'] + next_sub['output_toks']),
                'arrival_time_ns': release_time_ns,
                'session_id': session_id,
                'sub_request_index': next_idx,
            }
            if self._enable_prefix_caching:
                req_data['input_hash_ids'] = next_sub.get('input_tok_ids', [])
                req_data['output_hash_ids'] = next_sub.get('output_tok_ids', [])
            # Insert in sorted position after _pending_idx
            self._insert_pending_sorted(req_data)
            self._request_to_session[next_id] = (session_id, next_idx)
            session['next_index'] = next_idx + 1
        else:
            # Session complete; all sub-requests have been released.
            del self._deferred_sessions[session_id]

    def _insert_pending_sorted(self, req_data):
        """Insert a request in arrival-time order for the unconsumed suffix."""
        arrival = req_data['arrival_time_ns']
        # Binary search in the unconsumed portion
        lo = self._pending_idx
        hi = len(self._pending_requests)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._pending_requests[mid]['arrival_time_ns'] <= arrival:
                lo = mid + 1
            else:
                hi = mid
        self._pending_requests.insert(lo, req_data)

    def has_deferred_sessions(self):
        """Check if there are agentic sessions with unreleased sub-requests."""
        return bool(self._deferred_sessions)

    def get_next_pending_arrival(self):
        """Return the next pending request's arrival time, or None."""
        if self._pending_idx < len(self._pending_requests):
            return self._pending_requests[self._pending_idx]['arrival_time_ns']
        return None

    # -----------------------------------------------------------------------
    # P/D handoff, runtime migration hook, and completion helpers
    # -----------------------------------------------------------------------

    def transfer_prefill_request(self, requests):
        for req in requests:
            pool_id = req.pool_id
            if pool_id is None:
                pd_pools = [p for p in self.pools if p["mode"] == "pd"]
                if len(pd_pools) != 1:
                    raise RuntimeError("Prefill request is missing pool_id.")
                pool = pd_pools[0]
                req.pool_id = pool["id"]
            else:
                pool = self.pool_by_id.get(pool_id)
            if pool is None or pool["mode"] != "pd":
                raise RuntimeError(f"Request #{req.id} cannot transfer to decode pool '{pool_id}'.")

            sched = self._select_pool_scheduler(pool, "decode")
            req.route_history.append({
                "pool_id": pool["id"],
                "role": "decode",
                "instance_id": sched.instance_id,
            })
            sched.add_decode(req)

    def maybe_migrate_request(self, req, current_time_ns):
        return False

    def can_decode_instance_finish(self, instance_id):
        pool_id = self.instance_pool.get(instance_id)
        if pool_id is None:
            return True
        pool = self.pool_by_id[pool_id]
        if pool["mode"] != "pd":
            return True
        return not any(
            len(s.request) > 0 or len(s.inflight) > 0
            for s in pool["prefill_schedulers"]
        )

    # -----------------------------------------------------------------------
    # Legacy: upfront routing (kept for backward compat)
    # -----------------------------------------------------------------------

    def generate(self, path, enable_prefix_caching=False, is_init=True):
        """Load and immediately route all requests (legacy behavior)."""
        self.load_requests(path, enable_prefix_caching, is_init)
        # Route all at once (arrival time ignored)
        self.route_arrived_requests(float('inf'))
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type)",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )

    def transfer_prefill_request(self, requests, source_scheduler=None, current_time_ns=-1):
        for req in requests:
            instance_id = self._select_instance(self.decode_schedulers, "decode")
            self.decode_schedulers[instance_id].add_decode(
                req, source_scheduler=source_scheduler, current=current_time_ns)
