import hashlib
import redis as redis_lib
from config.settings import settings


def file_hash(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha.update(block)
    return sha.hexdigest()


def is_already_indexed(fhash: str, r: redis_lib.Redis) -> bool:
    return bool(r.sismember("indexed_hashes", fhash))


def mark_as_indexed(fhash: str, doc_id: str, r: redis_lib.Redis) -> None:
    r.sadd("indexed_hashes", fhash)
    r.hset("hash_to_doc_id", fhash, doc_id)


def get_doc_id_by_hash(fhash: str, r: redis_lib.Redis) -> str | None:
    val = r.hget("hash_to_doc_id", fhash)
    return val.decode() if val else None
