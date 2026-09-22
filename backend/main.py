import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "resolveAI_customer_support_dataset_v2.csv"
FAISS_PATH = BASE_DIR / "faiss_index"


# ============================================================
# CONFIG
# ============================================================

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ResolveAI API",
    description="Autonomous Customer Support & Resolution Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATASET
# ============================================================

if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

ID_COLUMNS = [
    "ticket_id",
    "customer_id",
    "order_id",
    "payment_id",
    "refund_id"
]

for column in ID_COLUMNS:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.upper()
            .str.strip()
        )


# ============================================================
# FAISS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

if FAISS_PATH.exists():
    vectorstore = FAISS.load_local(
        str(FAISS_PATH),
        embeddings,
        allow_dangerous_deserialization=True
    )


# ============================================================
# MONGODB
# ============================================================

cases_collection = None

if MongoClient is not None:
    try:
        mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=2000
        )

        mongo_client.admin.command("ping")

        db = mongo_client["resolveai"]
        cases_collection = db["cases"]

    except Exception:
        cases_collection = None


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str
    customer_id: Optional[str] = None
    ticket_id: Optional[str] = None


# ============================================================
# ID EXTRACTION
# ============================================================

ID_PATTERNS = {
    "ticket_id": r"\bT\d+\b",
    "customer_id": r"\bC\d+\b",
    "order_id": r"\bORD[-]?\d+\b",
    "payment_id": r"\bPAY[-]?\d+\b",
    "refund_id": r"\bREF[-]?\d+\b",
}


def extract_ids(text: str):
    result = {}

    for key, pattern in ID_PATTERNS.items():
        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        if matches:
            result[key] = matches[0].upper()

    return result


# ============================================================
# STRUCTURED LOOKUP
# ============================================================

def find_case(
    ticket_id=None,
    order_id=None,
    customer_id=None
):

    if ticket_id and "ticket_id" in df.columns:
        result = df[df["ticket_id"] == ticket_id.upper()]

        if not result.empty:
            return result.iloc[0].to_dict()

    if order_id and "order_id" in df.columns:
        result = df[df["order_id"] == order_id.upper()]

        if not result.empty:
            return result.iloc[0].to_dict()

    if customer_id and "customer_id" in df.columns:
        result = df[df["customer_id"] == customer_id.upper()]

        if not result.empty:
            return result.iloc[0].to_dict()

    return {}


def get_customer_history(customer_id):

    if not customer_id:
        return []

    if "customer_id" not in df.columns:
        return []

    result = df[
        df["customer_id"] == customer_id.upper()
    ]

    return result.to_dict(orient="records")


# ============================================================
# VECTOR RETRIEVAL
# ============================================================

def retrieve_similar_cases(query, k=5):

    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search(
        query,
        k=k
    )

    results = []

    for doc in docs:
        results.append({
            "ticket_id": doc.metadata.get("ticket_id"),
            "customer_id": doc.metadata.get("customer_id"),
            "category": doc.metadata.get("complaint_category"),
            "subcategory": doc.metadata.get("sub_category"),
            "resolution_status": doc.metadata.get("resolution_status"),
            "content": doc.page_content
        })

    return results


# ============================================================
# RELEVANT TEAM ASSIGNMENT
# ============================================================

TEAM_ROUTING = {
    "payment": {
        "team": "Payments & Billing",
        "person": "Aarav Sharma",
        "email": "payments.support@resolveai.demo",
        "phone": "+91 98765 43021"
    },
    "refund": {
        "team": "Payments & Billing",
        "person": "Aarav Sharma",
        "email": "payments.support@resolveai.demo",
        "phone": "+91 98765 43021"
    },
    "billing": {
        "team": "Payments & Billing",
        "person": "Aarav Sharma",
        "email": "payments.support@resolveai.demo",
        "phone": "+91 98765 43021"
    },
    "order": {
        "team": "Order Operations",
        "person": "Riya Mehta",
        "email": "orders.support@resolveai.demo",
        "phone": "+91 98765 43022"
    },
    "delivery": {
        "team": "Logistics & Delivery",
        "person": "Kabir Singh",
        "email": "logistics.support@resolveai.demo",
        "phone": "+91 98765 43023"
    },
    "router": {
        "team": "Technical Support",
        "person": "Neha Verma",
        "email": "technical.support@resolveai.demo",
        "phone": "+91 98765 43024"
    },
    "technical": {
        "team": "Technical Support",
        "person": "Neha Verma",
        "email": "technical.support@resolveai.demo",
        "phone": "+91 98765 43024"
    },
    "account": {
        "team": "Account & Security",
        "person": "Arjun Kapoor",
        "email": "account.support@resolveai.demo",
        "phone": "+91 98765 43025"
    },
    "subscription": {
        "team": "Subscriptions & Plans",
        "person": "Meera Joshi",
        "email": "plans.support@resolveai.demo",
        "phone": "+91 98765 43026"
    },
}


def assign_team(message, case):

    text = " ".join([
        message,
        str(case.get("complaint_category", "")),
        str(case.get("sub_category", "")),
        str(case.get("root_cause", "")),
    ]).lower()

    for keyword, assignment in TEAM_ROUTING.items():
        if keyword in text:
            return assignment

    return {
        "team": "Customer Support",
        "person": "Priya Nair",
        "email": "support@resolveai.demo",
        "phone": "+91 98765 43020"
    }


# ============================================================
# ACTION ENGINE
# ============================================================

def determine_action(message, case):

    text = message.lower()

    refund_status = str(
        case.get("refund_status", "")
    ).upper()

    if refund_status == "COMPLETED":
        return {
            "type": "REQUEST_FEEDBACK",
            "message": (
                "Your refund has already been completed. "
                "Please confirm whether you need any further assistance."
            )
        }

    if any(word in text for word in [
        "refund",
        "money back",
        "refund me",
        "want my money"
    ]):
        return {
            "type": "REFUND_APPROVAL",
            "message": (
                "I have investigated your request. "
                "A refund approval request has been initiated. "
                "A human support specialist must approve the refund "
                "before it can be processed."
            )
        }

    if any(word in text for word in [
        "payment",
        "charged",
        "declined",
        "transaction",
        "billing"
    ]):
        return {
            "type": "PAYMENT_INVESTIGATION",
            "message": (
                "I have investigated the payment-related issue "
                "using the available case and historical support data."
            )
        }

    if any(word in text for word in [
        "order",
        "cancelled",
        "order status"
    ]):
        return {
            "type": "ORDER_INVESTIGATION",
            "message": (
                "I have checked the order information and "
                "related historical cases."
            )
        }

    return {
        "type": "INVESTIGATE",
        "message": (
            "I have investigated your issue against the available "
            "customer-support knowledge base."
        )
    }


# ============================================================
# RESPONSE
# ============================================================

def generate_response(
    message,
    case,
    history,
    similar_cases,
    assignment,
    action
):

    if ChatGroq is not None and GROQ_API_KEY:

        try:
            llm = ChatGroq(
                model="openai/gpt-oss-120b",
                temperature=0
            )

            prompt = f"""
You are ResolveAI, an autonomous customer support agent.

Customer message:
{message}

Current case:
{case}

Customer history:
{history[:5]}

Similar historical cases:
{similar_cases[:3]}

Action:
{action}

Assigned specialist:
{assignment}

Answer naturally and clearly.

Rules:
- Use only available evidence.
- Do not invent transaction facts.
- Explain relevant investigation findings.
- If a refund requires approval, say that it is awaiting human approval.
- Never claim a real refund was processed unless the case data says so.
- Mention the relevant team/person when useful.
- This is a prototype, so operational actions are simulated.
"""

            result = llm.invoke(prompt)

            return result.content

        except Exception:
            pass

    response = action["message"]

    if case.get("root_cause"):
        response += (
            f"\n\n**Investigation finding:** "
            f"{case['root_cause']}."
        )

    if case.get("resolution_text"):
        response += (
            f"\n\n**Historical resolution context:** "
            f"{case['resolution_text']}"
        )

    response += (
        f"\n\n**Assigned team:** {assignment['team']}"
        f"\n**Assigned specialist:** {assignment['person']}"
        f"\n**Contact:** {assignment['email']} | {assignment['phone']}"
    )

    return response


# ============================================================
# SAVE CASE TO MONGODB
# ============================================================

def save_case(
    message,
    ids,
    case,
    assignment,
    action,
    similar_cases,
    response
):

    if cases_collection is None:
        return

    ticket_id = (
        ids.get("ticket_id")
        or case.get("ticket_id")
        or f"CHAT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )

    status = (
        "PENDING_APPROVAL"
        if action["type"] == "REFUND_APPROVAL"
        else "OPEN"
    )

    document = {
        "ticket_id": ticket_id,
        "customer_id": ids.get(
            "customer_id",
            case.get("customer_id")
        ),
        "issue": message,
        "status": status,
        "current_case": case,
        "assignment": assignment,
        "action": action,
        "similar_cases_count": len(similar_cases),
        "response": response,
        "updated_at": datetime.now(timezone.utc)
    }

    cases_collection.update_one(
        {"ticket_id": ticket_id},
        {
            "$set": document,
            "$push": {
                "timeline": {
                    "event": "AI_SUPPORT_RESPONSE",
                    "message": message,
                    "timestamp": datetime.now(timezone.utc)
                }
            }
        },
        upsert=True
    )


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "service": "ResolveAI",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "vector_db": vectorstore is not None,
        "mongodb": cases_collection is not None,
        "llm": ChatGroq is not None and bool(GROQ_API_KEY)
    }


@app.post("/chat")
def chat(request: ChatRequest):

    message = request.message.strip()

    ids = extract_ids(message)

    customer_id = (
        request.customer_id
        or ids.get("customer_id")
    )

    ticket_id = (
        request.ticket_id
        or ids.get("ticket_id")
    )

    order_id = ids.get("order_id")

    case = find_case(
        ticket_id=ticket_id,
        order_id=order_id,
        customer_id=customer_id
    )

    history = get_customer_history(customer_id)

    similar_cases = retrieve_similar_cases(
        message,
        k=5
    )

    assignment = assign_team(
        message,
        case
    )

    action = determine_action(
        message,
        case
    )

    response = generate_response(
        message=message,
        case=case,
        history=history,
        similar_cases=similar_cases,
        assignment=assignment,
        action=action
    )

    save_case(
        message=message,
        ids=ids,
        case=case,
        assignment=assignment,
        action=action,
        similar_cases=similar_cases,
        response=response
    )

    return {
        "response": response,
        "action": action["type"],
        "assigned_team": assignment["team"],
        "assigned_person": assignment["person"],
        "assigned_email": assignment["email"],
        "assigned_phone": assignment["phone"],
        "similar_cases_count": len(similar_cases),
        "customer_history_count": len(history),
        "case_found": bool(case)
    }
