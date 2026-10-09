### COMPLETE THE CODE  ###

from langchain_core.tools import Tool
from rag import retriever

def retrieve_policy(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found in the university IT policy document."
    return "\n\n".join([doc.page_content for doc in docs])

rag_tool = Tool(
    name="university_it_policy_search",
    description=(
        "Search the university IT policy document for relevant information. "
        "Use this when asked about IT support, Wi-Fi, VPN, printing, accounts, "
        "storage, email, or phishing. If the answer is not in the policy, "
        "state that the information is not available. Do NOT invent any facts."
    ),
    func=retrieve_policy,
)
