# SPDX-FileCopyrightText: Copyright 2026 routerx contributors
# SPDX-License-Identifier: Apache-2.0

"""컨테이너 진입점 — router-run 인터페이스를 그대로 전달한다."""
from __future__ import annotations

import sys

from routerx.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
