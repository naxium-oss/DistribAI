import sys
from pathlib import Path

# tools/trainer_translate/tests -> tools  (so `import trainer_translate` works)
TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
