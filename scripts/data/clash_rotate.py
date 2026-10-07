#!/usr/bin/env python3
"""ClashX node rotator: switch egress IPs programmatically to spread requests.

ClashX exposes a REST controller (pass its URL via --clash-controller);
PUT /proxies/<group> {"name": "<node>"} switches the selected node, which
changes the egress IP for every request through the local proxy.  Spreading
Wikipedia traffic across many nodes avoids per-IP rate limiting (429).

The original selection is restored via restore() when the job finishes.
"""
from __future__ import annotations

import json
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

SKIP_TOKENS = ("DIRECT", "REJECT", "选择节点", "升级", "官网", "客户端")


class NodeRotator:
    def __init__(self, controller=None, group="🔰 选择节点",
                 nodes=None, rotate_every=120, switch_wait=2.5, verbose=True):
        if not controller:
            raise ValueError("ClashX controller URL is required for node rotation "
                             "(pass --clash-controller, or disable rotation with --no-rotate)")
        self.controller = controller.rstrip("/")
        self.group = group
        self.rotate_every = max(1, rotate_every)
        self.switch_wait = switch_wait
        self.verbose = verbose
        self.counter = 0
        self.rotations = 0
        self.original = self._current()
        self.nodes = [n.strip() for n in nodes if n.strip()] if nodes else self._discover()
        if not self.nodes:
            raise RuntimeError("no usable nodes found in group %r" % group)
        self.index = 0
        if self.verbose:
            print("[rotator] nodes=%d current=%s" % (len(self.nodes), self.original))

    def _get(self, path):
        with urlopen(Request(self.controller + path, headers={"Accept": "application/json"}),
                     timeout=10) as response:
            return json.load(response)

    def _current(self):
        data = self._get("/proxies/" + quote(self.group))
        return data.get("now")

    def _discover(self):
        data = self._get("/proxies/" + quote(self.group))
        return [name for name in data.get("all", [])
                if name and not any(token in name for token in SKIP_TOKENS)]

    def switch(self, name):
        request = Request(self.controller + "/proxies/" + quote(self.group),
                          data=json.dumps({"name": name}).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="PUT")
        with urlopen(request, timeout=10) as response:
            if response.status not in (200, 204):
                raise RuntimeError("switch returned HTTP %s" % response.status)
        if self.switch_wait:
            time.sleep(self.switch_wait)

    def rotate(self):
        self.index = (self.index + 1) % len(self.nodes)
        node = self.nodes[self.index]
        try:
            self.switch(node)
            self.rotations += 1
            if self.verbose:
                print("[rotator] switched to %s (rotation #%d)" % (node, self.rotations), flush=True)
            return node
        except Exception as exc:  # noqa: BLE001 - keep crawling on a stale IP
            if self.verbose:
                print("[rotator] switch failed: %s" % exc, flush=True)
            return None

    def maybe_rotate(self):
        self.counter += 1
        if self.counter % self.rotate_every == 0:
            self.rotate()

    def restore(self):
        if self.original:
            try:
                self.switch(self.original)
                if self.verbose:
                    print("[rotator] restored %s" % self.original, flush=True)
            except Exception:
                pass
