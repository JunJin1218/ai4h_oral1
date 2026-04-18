"""
Claude teacher-eval entrypoint under assessor/.

This mirrors the location/usage style of assessor/evaluate_teacher_llm.py
while reusing the Claude implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from data_augmentation_claude.evaluate_teacher_llm_claude import main


if __name__ == "__main__":
    main()
