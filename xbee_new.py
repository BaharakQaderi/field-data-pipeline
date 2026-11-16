# ============================================================
# common/arbiter/packet_selector.py
# ============================================================
# Periodic selection of the "trusted" packet between MAIN/BACKUP.
# Inputs: two per-radio queues of parsed messages (dicts).
# Output: selected packet published to an output queue/callback.
# ============================================================
 
import asyncio
import time
from typing import Optional, Dict, Any
from .counters import Counters
 
MAIN = "MAIN"
BACKUP = "BACKUP"
 
class PacketSelector:
    """
    Event-driven packet selector with duplicate prevention.
   
    Architecture:
      1) Waits for at least one queue (MAIN or BACKUP) to have a packet
      2) Drains both queues to keep only the latest packet per link
      3) Validates packet counters > last_processed (prevents duplicates)
      4) Selects MAIN if valid, else BACKUP if valid, else marks lost
      5) Updates counters and publishes chosen packet to output queue
      6) Repeats immediately (no artificial delay, fully event-driven)
   
    Key features:
    - Event-driven: blocks until packets arrive, no CPU waste
    - No queue overflow: drains queues to keep only latest packet
    - Duplicate prevention: rejects already-processed counters
    - Latest-only semantics: maintains data freshness, discards stale packets
    - Priority-based selection: MAIN preferred over BACKUP
    """
 
    def __init__(
        self,
        queue_main: asyncio.Queue,          # parsed dicts from MAIN receiver
        queue_backup: asyncio.Queue,        # parsed dicts from BACKUP receiver
        output_queue: asyncio.Queue,        # downstream consumers (UI, logic)
        logger,
        period_sec: float = 0.05,           # 50 ms
        counter_field: str = "counter",
        preferred_link: str = MAIN,         # MAIN gets priority
    ):
        self.q_main = queue_main
        self.q_backup = queue_backup
        self.q_out = output_queue
        self.logger = logger
        self.period = period_sec
        self.counter_field = counter_field
        self.preferred_link = preferred_link
        self.counters = Counters()
        self.running = False
 
        # last packets drained each cycle (latest-only semantics)
        self._latest_main: Optional[Dict[str, Any]] = None
        self._latest_backup: Optional[Dict[str, Any]] = None
 
    # ---------- Core loop ----------
    async def run_loop(self):
        self.running = True
        self.logger.info(
            f"[Arbiter] Start event-driven selection loop "
            f"(counter='{self.counter_field}', preferred={self.preferred_link})"
        )
 
        while self.running:
            try:
                # 1) Drain both queues (keep only the latest packet per link)
                await self._drain_inputs()
 
                # 2) Validate packets using counter rule
                chosen = self._select_packet()
 
                # 3) Update counters & publish or mark lost
                if chosen is not None:
                    pkt_counter = chosen.get(self.counter_field, 0)
                    self.counters.mark_processed(pkt_counter)
 
                    # Mutate chosen packet directly
                    chosen["source"] = chosen.get("source")
                    chosen["counters"] = self.counters.snapshot()
                    chosen["decision_time"] = time.time()
 
                    try:
                        self.q_out.put_nowait(chosen)
                    except asyncio.QueueFull:
                        _ = self.q_out.get_nowait()
                        self.q_out.put_nowait(chosen)
                else:
                    self.counters.mark_lost()
                    self._latest_main = None
                    self._latest_backup = None
 
 
            except Exception as e:
                self.logger.error(f"[Arbiter] Error in selection loop: {e}")
 
 
    # ---------- Helpers ----------
    async def _drain_inputs(self):
        """Wait for at least one queue to have elements, then drain both to keep only latest."""
        # Wait for at least one queue to have an item
        main_task = asyncio.create_task(self.q_main.get())
        backup_task = asyncio.create_task(self.q_backup.get())
       
        done, pending = await asyncio.wait(
            {main_task, backup_task},
            return_when=asyncio.FIRST_COMPLETED
        )
       
        # Get the first item that arrived
        for task in done:
            item = task.result()
            if item in [t.result() for t in done if t == main_task]:
                if isinstance(item, dict):
                    item["source"] = MAIN
                    self._latest_main = item
            else:
                if isinstance(item, dict):
                    item["source"] = BACKUP
                    self._latest_backup = item
       
        # Cancel pending task if one is still waiting
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
       
        # Now drain any remaining items from both queues
        if self._latest_main is None:
            self._latest_main = await self._drain_one(self.q_main, MAIN)
        else:
            remaining = await self._drain_one(self.q_main, MAIN)
            if remaining:
                self._latest_main = remaining
               
        if self._latest_backup is None:
            self._latest_backup = await self._drain_one(self.q_backup, BACKUP)
        else:
            remaining = await self._drain_one(self.q_backup, BACKUP)
            if remaining:
                self._latest_backup = remaining
 
    async def _drain_one(self, q: asyncio.Queue, source_tag: str) -> Optional[Dict[str, Any]]:
        latest = None
        while not q.empty():
            try:
                item = q.get_nowait()
                if isinstance(item, dict):
                    item["source"] = source_tag
                latest = item
            except asyncio.QueueEmpty:
                break
        return latest
 
    def _is_valid(self, pkt: Optional[Dict[str, Any]]) -> bool:
        if not pkt:
            return False
        counter = pkt.get(self.counter_field)
        if counter is None:
            return False
        return counter > self.counters.last_processed
 
    def _select_packet(self) -> Optional[Dict[str, Any]]:
        """Apply the priority rule: MAIN first, else BACKUP, else None."""
        main_valid = self._is_valid(self._latest_main)
        back_valid = self._is_valid(self._latest_backup)
 
        chosen = None
        if self.preferred_link == MAIN:
            if main_valid:
                chosen = self._latest_main
            elif back_valid:
                chosen = self._latest_backup
        else:  # if ever inverted preference is needed
            if back_valid:
                chosen = self._latest_backup
            elif main_valid:
                chosen = self._latest_main
 
        # Optional: log decisions at debug level
        if chosen is not None:
            src = chosen.get("source")
            cnt = chosen.get(self.counter_field)
            self.logger.debug(f"[Arbiter] Selected {src} counter={cnt} "
                              f"(last_processed={self.counters.last_processed})")
 
        return chosen
 
    def stop(self):
        self.running = False
 
