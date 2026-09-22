import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient


# ============================================================
# PATHS / ENVIRONMENT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Local development: load .env from project root
# Render: environment variables are provided directly by Render
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env")

DATASET_PATH = BASE_DIR / "resolveAI_customer_support_dataset_v2.csv"
FAISS_PATH = BASE_DIR / "faiss_index"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017"
)

# IMPORTANT:
# This is the same model that was used to create the
# existing FAISS index.
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ResolveAI API",
    description="Autonomous AI Customer Support Backend",
    version="1.0.0"
)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

for column in [
    "ticket_id",
    "customer_id",
    "order_id",
    "payment_id",
    "refund_id"
]:
    if column in df.columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.upper()
            .str.strip()
        )


# ============================================================
# MONGODB
# ============================================================

client = None
db = None
cases_collection = None

try:
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=3000
    )

    db = client["resolveai"]
    cases_collection = db["cases"]

except Exception as e:

    print(f"MongoDB initialization failed: {e}")

    client = None
    db = None
    cases_collection = None


# ============================================================
# HUGGING FACE API EMBEDDINGS
# ============================================================

class HuggingFaceAPIEmbeddings(Embeddings):

    def __init__(
        self,
        api_key: str,
        model: str
    ):
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )

        self.model = model

    @staticmethod
    def _to_list(value):

        if hasattr(value, "tolist"):
            return value.tolist()

        return value

    def embed_query(
        self,
        text: str
    ) -> List[float]:

        embedding = self.client.feature_extraction(
            text,
            model=self.model
        )

        embedding = self._to_list(embedding)

        if (
            embedding
            and isinstance(embedding[0], list)
        ):
            embedding = embedding[0]

        return [
            float(x)
            for x in embedding
        ]

    def embed_documents(
        self,
        texts: List[str]
    ) -> List[List[float]]:

        results = []

        for text in texts:

            embedding = self.client.feature_extraction(
                text,
                model=self.model
            )

            embedding = self._to_list(embedding)

            if (
                embedding
                and isinstance(embedding[0], list)
            ):
                embedding = embedding[0]

            results.append([
                float(x)
                for x in embedding
            ])

        return results


# ============================================================
# LOAD FAISS
# ============================================================

vectorstore = None

if FAISS_PATH.exists() and HF_TOKEN:

    try:

        embeddings = HuggingFaceAPIEmbeddings(
            api_key=HF_TOKEN,
            model=HF_EMBEDDING_MODEL
        )

        vectorstore = FAISS.load_local(
            str(FAISS_PATH),
            embeddings,
            allow_dangerous_deserialization=True
        )

        print("FAISS index loaded successfully.")

    except Exception as e:

        print(
            f"FAISS initialization failed: {e}"
        )

        vectorstore = None

else:

    if not HF_TOKEN:
        print(
            "HF_TOKEN not found. "
            "FAISS vector search is disabled."
        )

    if not FAISS_PATH.exists():
        print(
            "FAISS index directory not found."
        )


# ============================================================
# GROQ
# ============================================================

llm = None

if GROQ_API_KEY:

    try:

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            api_key=GROQ_API_KEY
        )

        print("Groq LLM initialized.")

    except Exception as e:

        print(
            f"Groq initialization failed: {e}"
        )

        llm = None


# ============================================================
# ID EXTRACTION
# ============================================================

ID_PATTERNS = {

    "ticket_id":
        r"\bT\d+\b",

    "customer_id":
        r"\bC\d+\b",

    "order_id":
        r"\bORD[-]?\d+\b",

    "payment_id":
        r"\bPAY[-]?\d+\b",

    "refund_id":
        r"\bREF[-]?\d+\b"
}


def extract_ids(
    text: str
) -> Dict[str, List[str]]:

    found_ids = {}

    for id_type, pattern in ID_PATTERNS.items():

        matches = re.findall(
            pattern,
            text,
            re.IGNORECASE
        )

        if matches:

            found_ids[id_type] = [
                match.upper()
                for match in matches
            ]

    return found_ids


# ============================================================
# STRUCTURED LOOKUPS
# ============================================================

def find_by_id(
    id_type: str,
    id_value: Optional[str]
):

    if (
        not id_value
        or id_type not in df.columns
    ):
        return None

    result = df[
        df[id_type]
        == str(id_value)
        .upper()
        .strip()
    ]

    if result.empty:
        return None

    return result.iloc[0]


def get_customer_history(
    customer_id: Optional[str]
):

    if not customer_id:
        return None

    history = df[
        df["customer_id"]
        == customer_id.upper().strip()
    ]

    if history.empty:
        return None

    if "timestamp" in history.columns:

        return history.sort_values(
            "timestamp"
        )

    return history


# ============================================================
# ORGANIZATIONAL MEMORY / RAG
# ============================================================

def retrieve_organizational_memory(
    query: str,
    k: int = 5
) -> List[Dict[str, Any]]:

    if vectorstore is None:
        return []

    try:

        docs = vectorstore.similarity_search(
            query,
            k=k,
            fetch_k=50
        )

        results = []

        for doc in docs:

            results.append({

                "ticket_id":
                    doc.metadata.get(
                        "ticket_id"
                    ),

                "customer_id":
                    doc.metadata.get(
                        "customer_id"
                    ),

                "complaint_category":
                    doc.metadata.get(
                        "complaint_category"
                    ),

                "sub_category":
                    doc.metadata.get(
                        "sub_category"
                    ),

                "resolution_status":
                    doc.metadata.get(
                        "resolution_status"
                    ),

                "content":
                    doc.page_content
            })

        return results

    except Exception as e:

        print(
            f"RAG retrieval failed: {e}"
        )

        return []


# ============================================================
# TEAM ASSIGNMENT
# ============================================================

TEAM_ASSIGNMENTS = {

    "Payments & Billing": {
        "person": "Aarav Sharma",
        "email": "payments.support@resolveai.demo",
        "phone": "+91 98765 43021"
    },

    "Order Operations": {
        "person": "Riya Mehta",
        "email": "orders.support@resolveai.demo",
        "phone": "+91 98765 43022"
    },

    "Logistics & Delivery": {
        "person": "Kabir Singh",
        "email": "logistics.support@resolveai.demo",
        "phone": "+91 98765 43023"
    },

    "Technical Support": {
        "person": "Neha Verma",
        "email": "technical.support@resolveai.demo",
        "phone": "+91 98765 43024"
    },

    "Account & Security": {
        "person": "Arjun Kapoor",
        "email": "account.support@resolveai.demo",
        "phone": "+91 98765 43025"
    },

    "Subscriptions & Plans": {
        "person": "Meera Joshi",
        "email": "plans.support@resolveai.demo",
        "phone": "+91 98765 43026"
    },

    "Customer Support": {
        "person": "Priya Nair",
        "email": "support@resolveai.demo",
        "phone": "+91 98765 43020"
    }
}


def assign_team(
    case: Dict[str, Any],
    query: str
):

    text = (
        f"{query} "
        f"{case.get('complaint_category', '')} "
        f"{case.get('sub_category', '')}"
    ).lower()

    if any(
        word in text
        for word in [
            "payment",
            "refund",
            "billing",
            "charged",
            "card",
            "transaction"
        ]
    ):

        team = "Payments & Billing"

    elif any(
        word in text
        for word in [
            "order",
            "cancelled",
            "cancel",
            "purchase"
        ]
    ):

        team = "Order Operations"

    elif any(
        word in text
        for word in [
            "delivery",
            "shipping",
            "shipment",
            "courier"
        ]
    ):

        team = "Logistics & Delivery"

    elif any(
        word in text
        for word in [
            "router",
            "device",
            "technical",
            "crash",
            "restart"
        ]
    ):

        team = "Technical Support"

    elif any(
        word in text
        for word in [
            "account",
            "password",
            "login",
            "security",
            "suspicious",
            "unauthorized"
        ]
    ):

        team = "Account & Security"

    elif any(
        word in text
        for word in [
            "subscription",
            "plan",
            "upgrade",
            "downgrade"
        ]
    ):

        team = "Subscriptions & Plans"

    else:

        team = "Customer Support"

    assignment = TEAM_ASSIGNMENTS[team]

    return {

        "assigned_team":
            team,

        "assigned_person":
            assignment["person"],

        "assigned_email":
            assignment["email"],

        "assigned_phone":
            assignment["phone"]
    }


# ============================================================
# ACTION ENGINE
# ============================================================

def determine_action(
    case: Dict[str, Any],
    query: str
):

    refund_status = str(
        case.get(
            "refund_status",
            ""
        )
    ).upper()

    refund_requested = case.get(
        "refund_requested",
        False
    )

    order_status = str(
        case.get(
            "order_status",
            ""
        )
    ).upper()

    payment_status = str(
        case.get(
            "payment_status",
            ""
        )
    ).upper()

    if refund_status == "COMPLETED":

        return {

            "action":
                "REQUEST_FEEDBACK",

            "reason":
                "Refund has already been completed. "
                "No duplicate refund should be created. "
                "Customer confirmation is required."
        }

    if refund_requested:

        return {

            "action":
                "REFUND_REVIEW",

            "reason":
                "Customer has requested a refund. "
                "The case should proceed through refund approval."
        }

    if order_status == "CANCELLED":

        return {

            "action":
                "ORDER_INVESTIGATION",

            "reason":
                "The order is cancelled and requires "
                "investigation of the order/payment state."
        }

    if payment_status == "FAILED":

        return {

            "action":
                "PAYMENT_INVESTIGATION",

            "reason":
                "The payment is marked failed and requires "
                "payment investigation."
        }

    return {

        "action":
            "CUSTOMER_ASSISTANCE",

        "reason":
            "No specific transactional action is required "
            "from the available structured data."
    }


# ============================================================
# GROQ RESPONSE
# ============================================================

def generate_response(
    query: str,
    current_case: Dict[str, Any],
    customer_history: List[Dict[str, Any]],
    similar_cases: List[Dict[str, Any]],
    action: Dict[str, Any]
) -> str:

    fallback = (
        f"I investigated your request. "
        f"{action['reason']} "
        f"Based on the available case information, "
        f"the next step is {action['action']}."
    )

    if llm is None:
        return fallback

    try:

        prompt = f"""
You are ResolveAI, an autonomous customer support agent.

Answer the customer's request using only the supplied evidence.
Do not invent transaction details, refunds, approvals, or actions.

Customer request:
{query}

Current case:
{current_case}

Customer history:
{customer_history[:10]}

Similar resolved organizational cases:
{similar_cases[:5]}

System action decision:
{action}

Give a concise, professional response.
Explain what was found and what the next step is.
If a refund is already completed, explicitly say that a
duplicate refund will not be created.
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        print(
            f"Groq response generation failed: {e}"
        )

        return fallback


# ============================================================
# MONGODB CASE STORAGE
# ============================================================

def save_case(
    case_document: Dict[str, Any]
):

    if cases_collection is None:
        return False

    try:

        ticket_id = case_document.get(
            "ticket_id"
        )

        if ticket_id:

            cases_collection.update_one(

                {
                    "ticket_id":
                        ticket_id
                },

                {
                    "$set":
                        case_document
                },

                upsert=True
            )

        else:

            cases_collection.insert_one(
                case_document
            )

        return True

    except Exception as e:

        print(
            f"MongoDB save failed: {e}"
        )

        return False


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    message: str

    customer_id: Optional[str] = None

    ticket_id: Optional[str] = None


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def root():

    return {

        "service":
            "ResolveAI",

        "status":
            "running"
    }


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "vector_db":
            vectorstore is not None,

        "mongodb":
            cases_collection is not None,

        "llm":
            llm is not None,

        "hf_embeddings":
            bool(HF_TOKEN)
    }


@app.post("/chat")
def chat(
    request: ChatRequest
):

    query = request.message

    ids = extract_ids(query)

    customer_id = (
        request.customer_id
        or ids.get(
            "customer_id",
            [None]
        )[0]
    )

    ticket_id = (
        request.ticket_id
        or ids.get(
            "ticket_id",
            [None]
        )[0]
    )

    order_id = ids.get(
        "order_id",
        [None]
    )[0]


    # --------------------------------------------------------
    # CUSTOMER HISTORY
    # --------------------------------------------------------

    customer_history = []

    if customer_id:

        history = get_customer_history(
            customer_id
        )

        if history is not None:

            customer_history = (
                history
                .to_dict(
                    orient="records"
                )
            )


    # --------------------------------------------------------
    # CURRENT CASE
    # --------------------------------------------------------

    current_case = {}

    if ticket_id:

        result = find_by_id(
            "ticket_id",
            ticket_id
        )

        if result is not None:

            current_case = (
                result.to_dict()
            )

    elif order_id:

        result = find_by_id(
            "order_id",
            order_id
        )

        if result is not None:

            current_case = (
                result.to_dict()
            )


    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    similar_cases = (
        retrieve_organizational_memory(
            query,
            k=5
        )
    )


    # --------------------------------------------------------
    # ACTION
    # --------------------------------------------------------

    action = determine_action(
        current_case,
        query
    )


    # --------------------------------------------------------
    # TEAM ASSIGNMENT
    # --------------------------------------------------------

    assignment = assign_team(
        current_case,
        query
    )


    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    response_text = generate_response(

        query=query,

        current_case=current_case,

        customer_history=customer_history,

        similar_cases=similar_cases,

        action=action
    )


    # --------------------------------------------------------
    # SAVE CASE
    # --------------------------------------------------------

    case_document = {

        "ticket_id":
            ticket_id,

        "customer_id":
            customer_id,

        "issue":
            query,

        "current_case":
            current_case,

        "status":
            "WAITING_FOR_CUSTOMER",

        "action":
            action,

        "assignment":
            assignment,

        "customer_feedback": {

            "status":
                None,

            "message":
                None
        }
    }

    save_case(
        case_document
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "response":
            response_text,

        "action":
            action["action"],

        "action_reason":
            action["reason"],

        **assignment,

        "similar_cases_count":
            len(similar_cases),

        "customer_history_count":
            len(customer_history),

        "case_found":
            bool(current_case)
    }