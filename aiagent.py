"""
=====================================================================
 Gemini Tool-Calling Agent (built from scratch - no LangChain)
=====================================================================
 Model : gemini-3.1-flash-lite  (Gemini REST API, native function calling)

 Tools :
   1. get_gold_price   -> Real-time gold price (no API key)
   2. get_weather      -> Real-time weather by city (no API key)
   3. create_pdf       -> Convert raw data into a PDF file
   4. create_json      -> Convert raw data into a JSON file
   5. send_email       -> Send email via SMTP, with or without attachments

 .env file (same folder as this script):
   GEMINI_API_KEY=your_gemini_key
   SMTP_EMAIL=youremail@gmail.com
   SMTP_PASSWORD=your_app_password
   SMTP_SERVER=smtp.gmail.com        # optional (default: smtp.gmail.com)
   SMTP_PORT=587                     # optional (default: 587)

 Install:
   pip install requests reportlab

 Run:
   python gemini_agent.py
=====================================================================
"""

import os
import json
import smtplib
import mimetypes
from datetime import datetime
from email.message import EmailMessage
from urllib.parse import quote

import requests
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

def load_env(path=".env"):
    """Minimal .env loader (no python-dotenv needed)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.split(" #")[0].strip().strip('"').strip("'")
            os.environ.setdefault(key.strip(), value)


load_env()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SMTP_EMAIL = os.getenv("SMTP_EMAIL")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

MODEL_NAME = "gemini-3.1-flash-lite"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

OUTPUT_DIR = "outputs"
MAX_AGENT_STEPS = 8                 # safety limit for tool-call loops
ASK_BEFORE_SENDING_EMAIL = True     # set False to send emails without confirmation

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY not found in .env file.")


# =====================================================================
# 2. TOOL IMPLEMENTATIONS
# =====================================================================

def get_gold_price(currency="INR"):
    """Real-time gold spot price. Source: gold-api.com (USD) + open.er-api.com (FX)."""
    try:
        currency = (currency or "INR").upper()

        gold = requests.get("https://api.gold-api.com/price/XAU", timeout=10)
        gold.raise_for_status()
        gold_data = gold.json()
        usd_per_ounce = float(gold_data["price"])

        rate = 1.0
        if currency != "USD":
            fx = requests.get("https://open.er-api.com/v6/latest/USD", timeout=10)
            fx.raise_for_status()
            rates = fx.json().get("rates", {})
            if currency not in rates:
                return {"error": f"Currency '{currency}' is not supported."}
            rate = float(rates[currency])

        GRAMS_PER_TROY_OUNCE = 31.1034768
        price_per_ounce = usd_per_ounce * rate
        price_per_gram_24k = price_per_ounce / GRAMS_PER_TROY_OUNCE

        return {
            "metal": "Gold (XAU)",
            "currency": currency,
            "price_per_troy_ounce": round(price_per_ounce, 2),
            "price_per_gram_24k": round(price_per_gram_24k, 2),
            "price_per_10_gram_24k": round(price_per_gram_24k * 10, 2),
            "price_per_10_gram_22k": round(price_per_gram_24k * 10 * 22 / 24, 2),
            "usd_per_troy_ounce": round(usd_per_ounce, 2),
            "updated_at": gold_data.get("updatedAt"),
            "note": "International spot price. Local retail prices include duties/taxes and may differ."
        }
    except Exception as e:
        return {"error": f"Unable to get gold price: {e}"}


def get_weather(city):
    """Real-time weather for a city. Source: wttr.in (no API key)."""
    try:
        response = requests.get(
            f"https://wttr.in/{quote(city)}",
            params={"format": "j1"},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        current = data["current_condition"][0]

        return {
            "city": city,
            "temperature": f"{current['temp_C']} °C",
            "feels_like": f"{current['FeelsLikeC']} °C",
            "condition": current["weatherDesc"][0]["value"],
            "humidity": f"{current['humidity']}%",
            "wind_speed": f"{current['windspeedKmph']} km/h",
            "observation_time_utc": current.get("observation_time")
        }
    except Exception as e:
        return {"error": f"Unable to get weather: {e}"}


def _safe_filename(name, extension):
    """Clean the file name and add a timestamp if none given."""
    if not name:
        name = f"file_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
    if not name.lower().endswith(extension):
        name += extension
    return os.path.join(OUTPUT_DIR, name)


def create_pdf(content, title="Report", file_name=None):
    """Convert raw text into a PDF file."""
    try:
        file_path = _safe_filename(file_name or title, ".pdf")
        styles = getSampleStyleSheet()

        def escape(text):
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        story = [
            Paragraph(escape(title), styles["Title"]),
            Paragraph(f"Generated on {datetime.now().strftime('%d %b %Y, %I:%M %p')}", styles["Italic"]),
            Spacer(1, 0.6 * cm)
        ]

        for line in str(content).split("\n"):
            if line.strip():
                story.append(Paragraph(escape(line), styles["BodyText"]))
            else:
                story.append(Spacer(1, 0.3 * cm))

        doc = SimpleDocTemplate(
            file_path, pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm
        )
        doc.build(story)

        return {"status": "success", "file_path": os.path.abspath(file_path)}
    except Exception as e:
        return {"error": f"Unable to create PDF: {e}"}


def create_json(data, file_name=None):
    """Convert raw data into a JSON file. Accepts a JSON string or plain text."""
    try:
        file_path = _safe_filename(file_name, ".json")

        if isinstance(data, (dict, list)):
            parsed = data
        else:
            try:
                parsed = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                parsed = {"content": str(data)}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=4, ensure_ascii=False)

        return {"status": "success", "file_path": os.path.abspath(file_path)}
    except Exception as e:
        return {"error": f"Unable to create JSON: {e}"}


def send_email(to, subject, body, attachments=None, is_html=False):
    """Send an email via SMTP with optional attachments."""
    try:
        if not SMTP_EMAIL or not SMTP_PASSWORD:
            return {"error": "SMTP_EMAIL or SMTP_PASSWORD missing in .env file."}

        recipients = [r.strip() for r in str(to).replace(";", ",").split(",") if r.strip()]
        attachments = attachments or []

        missing = [a for a in attachments if not os.path.exists(a)]
        if missing:
            return {"error": f"Attachment(s) not found: {missing}"}

        if ASK_BEFORE_SENDING_EMAIL:
            print("\n----------- EMAIL PREVIEW -----------")
            print("To         :", ", ".join(recipients))
            print("Subject    :", subject)
            print("Attachments:", attachments if attachments else "None")
            print("Body       :\n", body)
            print("-------------------------------------")
            if input("Send this email? (y/n): ").strip().lower() != "y":
                return {"status": "cancelled", "message": "User cancelled sending the email."}

        msg = EmailMessage()
        msg["From"] = SMTP_EMAIL
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject

        if is_html:
            msg.set_content("This email contains HTML content.")
            msg.add_alternative(body, subtype="html")
        else:
            msg.set_content(body)

        for path in attachments:
            mime_type, _ = mimetypes.guess_type(path)
            maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
            with open(path, "rb") as f:
                msg.add_attachment(f.read(), maintype=maintype, subtype=subtype,
                                   filename=os.path.basename(path))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.send_message(msg)

        return {"status": "success", "sent_to": recipients, "attachments": attachments}
    except Exception as e:
        return {"error": f"Unable to send email: {e}"}


# =====================================================================
# 3. TOOL REGISTRY + DECLARATIONS (what Gemini sees)
# =====================================================================

TOOL_FUNCTIONS = {
    "get_gold_price": get_gold_price,
    "get_weather": get_weather,
    "create_pdf": create_pdf,
    "create_json": create_json,
    "send_email": send_email,
}

TOOL_DECLARATIONS = [
    {
        "name": "get_gold_price",
        "description": "Get the real-time international gold spot price per troy ounce, per gram and per 10 grams (24k and 22k).",
        "parameters": {
            "type": "object",
            "properties": {
                "currency": {
                    "type": "string",
                    "description": "3-letter currency code, e.g. INR, USD, AED, SGD. Default INR."
                }
            }
        }
    },
    {
        "name": "get_weather",
        "description": "Get the real-time current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Hyderabad"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "create_pdf",
        "description": "Convert raw text/data into a PDF file. Returns the saved file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Text content of the PDF. Use new lines for separate lines."},
                "title": {"type": "string", "description": "Title shown at the top of the PDF."},
                "file_name": {"type": "string", "description": "Optional file name without extension."}
            },
            "required": ["content"]
        }
    },
    {
        "name": "create_json",
        "description": "Convert raw data into a JSON file. Pass well-structured JSON as a string. Returns the saved file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "data": {"type": "string", "description": "Data as a valid JSON string (preferred) or plain text."},
                "file_name": {"type": "string", "description": "Optional file name without extension."}
            },
            "required": ["data"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email with or without attachments. For attachments, use file paths returned by create_pdf or create_json, or paths given by the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email(s), comma-separated."},
                "subject": {"type": "string", "description": "Email subject."},
                "body": {"type": "string", "description": "Email body."},
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of file paths to attach."
                },
                "is_html": {"type": "boolean", "description": "True if the body is HTML. Default false."}
            },
            "required": ["to", "subject", "body"]
        }
    }
]

SYSTEM_INSTRUCTION = """
You are a helpful AI agent with access to tools.

Rules:
1. Read the user's request and decide which tool(s) are needed.
2. If no tool is needed, answer directly.
3. If multiple steps are needed, call tools in the correct order.
   Example: "Email me today's gold price as PDF" ->
   get_gold_price -> create_pdf -> send_email (attach the returned file_path).
4. Always use the exact file_path returned by create_pdf / create_json as the attachment.
5. Never invent data. Use only tool results.
6. If required information is missing (e.g. recipient email or city), ask the user.
7. After the tools finish, give a short, clear final answer to the user.
"""


# =====================================================================
# 4. GEMINI API CALL (raw REST - no SDK, no LangChain)
# =====================================================================

def call_gemini(contents):
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": contents,
        "tools": [{"functionDeclarations": TOOL_DECLARATIONS}],
        "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}}
    }
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    response = requests.post(GEMINI_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

    data = response.json()
    candidates = data.get("candidates", [])
    if not candidates or "content" not in candidates[0]:
        raise RuntimeError(f"Empty response from Gemini: {data}")

    return candidates[0]["content"]


# =====================================================================
# 5. AGENT LOOP (decide -> execute tool -> send result back -> repeat)
# =====================================================================

def execute_tool(name, args):
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**args)
    except TypeError as e:
        return {"error": f"Invalid arguments for {name}: {e}"}


def run_agent(user_message, history):
    history.append({"role": "user", "parts": [{"text": user_message}]})

    for _ in range(MAX_AGENT_STEPS):
        model_content = call_gemini(history)

        # Keep the full model content (includes thought signatures required by Gemini 3)
        history.append(model_content)

        parts = model_content.get("parts", [])
        function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

        # No tool call -> final answer
        if not function_calls:
            return "".join(p.get("text", "") for p in parts).strip() or "(No response)"

        # Execute every tool Gemini selected
        response_parts = []
        for call in function_calls:
            name = call["name"]
            args = call.get("args", {}) or {}

            print(f"\n🔧 Tool selected : {name}")
            print(f"   Arguments     : {json.dumps(args, ensure_ascii=False)}")

            result = execute_tool(name, args)
            print(f"   Result        : {json.dumps(result, ensure_ascii=False)[:300]}")

            function_response = {"name": name, "response": {"result": result}}
            if "id" in call:
                function_response["id"] = call["id"]
            response_parts.append({"functionResponse": function_response})

        # Send tool results back to Gemini
        history.append({"role": "user", "parts": response_parts})

    return "Stopped: maximum agent steps reached."


# =====================================================================
# 6. MAIN - INTERACTIVE CHAT
# =====================================================================

def main():
    print("=" * 60)
    print(f" Gemini Agent ({MODEL_NAME})")
    print(" Tools: gold price | weather | PDF | JSON | email")
    print(" Type 'exit' to quit, 'reset' to clear conversation")
    print("=" * 60)

    history = []

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            history = []
            print("Conversation cleared.")
            continue

        try:
            answer = run_agent(user_input, history)
            print(f"\nAgent: {answer}")
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
