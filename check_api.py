"""One tiny Claude call that prints exactly what is wrong, if anything.  Run: python check_api.py"""
import os
import sys

import anthropic

import config  # noqa: F401  (loads .env)

key = os.environ.get("ANTHROPIC_API_KEY", "")
if not key:
    sys.exit("ANTHROPIC_API_KEY is not set in THIS terminal. Run: export ANTHROPIC_API_KEY=sk-ant-...")
print(f"key found: {key[:10]}...{key[-4:]} ({len(key)} chars)")
model = sys.argv[1] if len(sys.argv) > 1 else "claude-opus-5-5"
try:
    r = anthropic.Anthropic(max_retries=0, timeout=60).messages.create(
        model=model, max_tokens=50, messages=[{"role": "user", "content": "Reply with just: OK"}])
    print(f"SUCCESS ({model}): {r.content[-1].text!r}  tokens in/out {r.usage.input_tokens}/{r.usage.output_tokens}")
except anthropic.AuthenticationError as e:
    print("401 AUTH: key is wrong/revoked, or has a stray space/quote.", e.message)
except anthropic.PermissionDeniedError as e:
    print("403 PERMISSION: key's workspace can't use this model.", e.message)
except anthropic.NotFoundError as e:
    print(f"404: model {model!r} not available to this key.", e.message)
except anthropic.RateLimitError as e:
    print("429: rate limited, or the account has no credits loaded yet.", e.message)
except anthropic.BadRequestError as e:
    print("400:", e.message, "(a 'credit balance is too low' message here means credits haven't been added)")
except anthropic.APITimeoutError:
    print("TIMEOUT: reached the server but no reply within 60s.")
except anthropic.APIConnectionError as e:
    print("CONNECTION: could not reach api.anthropic.com (proxy/VPN/firewall?).", e)
except anthropic.APIStatusError as e:
    print(f"{e.status_code}:", e.message)
