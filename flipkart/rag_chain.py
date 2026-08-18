from langchain_groq import ChatGroq
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.runnables.history import (
    RunnableWithMessageHistory
)
from langchain_community.chat_message_histories import (
    ChatMessageHistory
)
from langchain_core.chat_history import (
    BaseChatMessageHistory
)

from flipkart.config import Config


class RAGChainBuilder:

    def __init__(self, vector_store):

        self.vector_store = vector_store

        # LLM used for question rewriting and final answers
        self.model = ChatGroq(
            model=Config.RAG_MODEL,
            temperature=0.2
        )

        # Stores conversation history for each session
        self.history_store = {}


    def _get_history(
        self,
        session_id: str
    ) -> BaseChatMessageHistory:

        if session_id not in self.history_store:
            self.history_store[session_id] = ChatMessageHistory()

        return self.history_store[session_id]


    def build_chain(self):

        # --------------------------------------------------
        # 1. Create Retriever
        # --------------------------------------------------

        retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 3
            }
        )


        # --------------------------------------------------
        # 2. Question Rewriting Prompt
        # --------------------------------------------------

        context_prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
You are a question rewriting assistant.

Your job is to convert the user's latest question into a
standalone question that can be understood without the
previous conversation.

Use the conversation history only when necessary.

Rules:

1. Do NOT answer the question.
2. Do NOT add new information.
3. Do NOT change the user's intention.
4. If the question is already standalone, keep it almost
   exactly the same.
5. If the question refers to something from the previous
   conversation using words such as:
   "it", "this", "that", "they", "their", "the product",
   or similar references, use the conversation history to
   make the question standalone.
6. Return ONLY the rewritten question.
"""
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{input}"
            )

        ])


        # --------------------------------------------------
        # 3. Final Answer Prompt
        # --------------------------------------------------

        qa_prompt = ChatPromptTemplate.from_messages([

            (
                "system",
                """
You are a professional e-commerce product recommendation
assistant.

Answer the user's question using ONLY the product reviews
and product titles provided below.

=========================
STRICT RULES
=========================

1. Use ONLY information available in the provided product
   information.

2. NEVER invent:
   - product names
   - prices
   - ratings
   - features
   - specifications
   - battery information
   - review information

3. If the requested information is not available, respond:

"I couldn't find that information in the available product reviews."

4. Never mention:
   - context
   - retrieved documents
   - vector database
   - embeddings
   - RAG
   - retrieval

5. Do not repeat the user's question.

6. Keep the answer concise, clear, and professional.

7. Use Markdown formatting where useful.

8. Do not create unnecessary sections.

9. Only include information that is actually available.

10. If multiple products are discussed, clearly separate
    each product.

11. Give a recommendation ONLY when the available reviews
    provide enough evidence.

12. Never claim that a product is better unless the reviews
    support that conclusion.

13. Do not combine information from different products
    incorrectly.

14. If information is available for only some products,
    answer only using the available information.


=========================
OUTPUT FORMATTING
=========================

### WHEN THE USER ASKS TO LIST PRODUCTS

Use:

### Products Found

1. **Product Name**
   - Feature: ...
   - Quality: ...
   - Battery: ...
   - Comfort: ...

2. **Product Name**
   - Feature: ...
   - Quality: ...
   - Battery: ...
   - Comfort: ...

Only include fields for which information is available.

Do NOT create empty fields.


=========================

### WHEN THE USER ASKS ABOUT A SPECIFIC FEATURE

Use:

### Feature Name

**Product Name**
- Relevant information from the reviews.

**Product Name**
- Relevant information from the reviews.

### Recommendation

Give a short recommendation only if the reviews support it.


=========================

### WHEN THE USER ASKS TO COMPARE PRODUCTS

Use:

### Product Comparison

**Product A**
- Advantage: ...
- Limitation: ...

**Product B**
- Advantage: ...
- Limitation: ...

### Recommendation

Briefly explain which product appears better based ONLY
on the available reviews.


=========================

### WHEN INFORMATION IS NOT AVAILABLE

Say:

I couldn't find that information in the available product reviews.


=========================
PRODUCT INFORMATION
=========================

{context}


=========================
USER QUESTION
=========================

{input}
"""
            ),

            MessagesPlaceholder(
                variable_name="chat_history"
            ),

            (
                "human",
                "{input}"
            )

        ])


        # --------------------------------------------------
        # 4. Create History-Aware Retriever
        # --------------------------------------------------

        history_aware_retriever = create_history_aware_retriever(
            self.model,
            retriever,
            context_prompt
        )


        # --------------------------------------------------
        # 5. Create Question Answer Chain
        # --------------------------------------------------

        question_answer_chain = create_stuff_documents_chain(
            self.model,
            qa_prompt
        )


        # --------------------------------------------------
        # 6. Create Complete RAG Chain
        # --------------------------------------------------

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain
        )


        # --------------------------------------------------
        # 7. Add Conversation History
        # --------------------------------------------------

        return RunnableWithMessageHistory(
            rag_chain,
            self._get_history,

            input_messages_key="input",

            history_messages_key="chat_history",

            output_messages_key="answer"
        )