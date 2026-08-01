"""Merkle credit-ledger root hash and inclusion-proof endpoints."""

from aiohttp import web

from worker.src.daemon.credit_ledger import CreditLedger


class LedgerHandler:
    """Read ledger root material and verify individual records."""

    def __init__(self, credit_ledger: CreditLedger) -> None:
        self.credit_ledger = credit_ledger

    async def get_root(self, req: web.Request) -> web.Response:
        """Current root hash, entry count, and optional signature bytes."""
        return web.json_response(
            {
                "root_hash": self.credit_ledger.get_root_hash().hex(),
                "size": self.credit_ledger.size(),
                "signature": self.credit_ledger.signature.hex()
                if self.credit_ledger.signature
                else None,
            }
        )

    async def verify_record(self, req: web.Request) -> web.Response:
        """Single ledger index plus Merkle proof and chain integrity flag."""
        try:
            index = int(req.match_info["index"])
        except (KeyError, ValueError):
            return web.json_response({"error": "invalid index"}, status=400)

        record = self.credit_ledger.get_record(index)
        if not record:
            return web.json_response({"error": "not found"}, status=404)

        proof = self.credit_ledger.get_proof(index)
        return web.json_response(
            {
                "index": record.index,
                "timestamp": record.timestamp,
                "hash": record.hash.hex(),
                "proof": [p.hex() for p in proof] if proof else [],
                "valid": self.credit_ledger.verify_chain_integrity(),
            }
        )
