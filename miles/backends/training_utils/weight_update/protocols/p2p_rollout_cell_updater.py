import logging
from argparse import Namespace
from collections.abc import Callable
from concurrent.futures import Future
from threading import Lock

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update import checksum_utils
from miles.backends.training_utils.weight_update.protocols.transports.mooncake import RemoteShard
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def __init__(self, args: Namespace, cell_id: str, api_client: SGLangApiClient) -> None:
        super().__init__(args=args, cell_id=cell_id, api_client=api_client)
        self._pending_writes: set[Future[None]] = set()
        self._pending_writes_lock = Lock()

    def run_write(
        self,
        write: Callable[[], None],
        *,
        remote_shard: RemoteShard,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        try:
            write()
            if sent_checksums is not None:
                self._verify_transfer_checksums(remote_shard=remote_shard, sent_checksums=sent_checksums)
        except Exception as error:
            self.mark_errored(error)
            raise

    def track_write(self, future: Future[None]) -> None:
        with self._pending_writes_lock:
            self._pending_writes.add(future)
        future.add_done_callback(self._forget_write)
        if self.is_errored:
            future.cancel()

    def mark_errored(self, error: BaseException) -> None:
        super().mark_errored(error)
        with self._pending_writes_lock:
            pending = tuple(self._pending_writes)
        for future in pending:
            future.cancel()

    def _forget_write(self, future: Future[None]) -> None:
        with self._pending_writes_lock:
            self._pending_writes.discard(future)

    def _verify_transfer_checksums(self, *, remote_shard: RemoteShard, sent_checksums: dict[str, str]) -> None:
        engine_body = self.submit_client_call(
            "check_weights", action="raw_checksum", names=list(sent_checksums), selector=remote_shard.runner_role
        ).result()
        if engine_body is None:
            return
        checksum_prefix = "" if remote_shard.runner_role == "target" else f"{remote_shard.runner_role}."
        checksum_utils.verify_transfer_checksums(
            sent_checksums={f"{checksum_prefix}{name}": checksum for name, checksum in sent_checksums.items()},
            engine_body=engine_body,
            cell_id=self.cell_id,
            rank=remote_shard.rollout_engine_rank,
        )
