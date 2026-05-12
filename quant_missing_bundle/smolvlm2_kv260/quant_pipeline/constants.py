from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOLVLM_LOCAL_DEFAULT = Path("/Users/miracle/smolvlm_local")
VERSAVLM_ROOT = Path("/Users/miracle/Documents/AICAS_more/VersaVLM")
FROZEN_EVAL_JSON = VERSAVLM_ROOT / "payload" / "data" / "sampled_100.json"
FROZEN_EVAL_IMAGES = VERSAVLM_ROOT / "payload" / "data" / "images"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "quant_experiments"
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cache"
RESULT_SCHEMA_PATH = PROJECT_ROOT / "quant_pipeline" / "result_schema.json"
LOCAL_WIKITEXT_CACHE = PROJECT_ROOT / "quant_pipeline" / "cache" / "wikitext_test.txt"

DEFAULT_SEED = 17
DEFAULT_DEVICE_PRIORITY = ("cuda", "mps", "cpu")
DEFAULT_QAT_TRAIN_SAMPLES = 5000
DEFAULT_QAT_VAL_SAMPLES = 500
DEFAULT_PTQ_MM_CALIB_SAMPLES = 512
DEFAULT_PTQ_TEXT_CALIB_SAMPLES = 256

TEXT_PPL_MAX_RATIO = 1.15
MULTIMODAL_MAX_DROP = 5.0

TEXT_ONLY_SANITY_PROMPTS = (
    "Summarize the purpose of quantization for edge inference in one sentence.",
    "List two risks of aggressive 4-bit quantization on a multimodal model.",
)
